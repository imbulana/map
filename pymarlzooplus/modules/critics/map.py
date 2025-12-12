from functools import wraps

import torch as th
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn import functional as F


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module


def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)


def cache_fn(f):
    cache = None

    @wraps(f)
    def cached_fn(*args, _cache=True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)

        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache

    return cached_fn


def dropout_seq(seq, mask, dropout):
    b, n = seq.shape[:2]
    device = seq.device
    logits = th.randn(b, n, device=device)

    if exists(mask):
        logits = logits.masked_fill(~mask, -th.finfo(logits.dtype).max)

    keep_prob = 1.0 - dropout
    num_keep = max(1, int(keep_prob * n))

    keep_indices = logits.topk(num_keep, dim=1).indices
    batch_indices = th.arange(b, device=device)
    batch_indices = rearrange(batch_indices, "b -> b 1")

    seq = seq[batch_indices, keep_indices]

    if exists(mask):
        seq_counts = mask.sum(dim=-1)
        seq_keep_counts = th.ceil(seq_counts * keep_prob).int()
        keep_mask = th.arange(num_keep, device=device) < rearrange(seq_keep_counts, "b -> b 1")
        mask = mask[batch_indices, keep_indices] & keep_mask

    return seq, mask


class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs["context"]
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)

        return self.fn(x, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None, mask=None):
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        sim = th.einsum("b i d, b j d -> b i j", q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -th.finfo(sim.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            sim.masked_fill_(~mask, max_neg_value)

        attn = sim.softmax(dim=-1)
        out = th.einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


class PerceiverIO(nn.Module):
    def __init__(
        self,
        *,
        depth,
        dim,
        queries_dim,
        logits_dim=None,
        num_latents=512,
        latent_dim=512,
        cross_heads=1,
        latent_heads=8,
        cross_dim_head=64,
        latent_dim_head=64,
        weight_tie_layers=False,
        decoder_ff=False,
        seq_dropout_prob=0.0,
    ):
        super().__init__()
        self.seq_dropout_prob = seq_dropout_prob
        self.latents = nn.Parameter(th.randn(num_latents, latent_dim))

        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(
                latent_dim, 
                Attention(latent_dim, dim, heads=cross_heads, dim_head=cross_dim_head), context_dim=dim),
            PreNorm(latent_dim, FeedForward(latent_dim))
        ])

        get_latent_attn = lambda: PreNorm(
            latent_dim, Attention(latent_dim, heads=latent_heads, dim_head=latent_dim_head)
        )
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        cache_args = {"_cache": weight_tie_layers}
        for _ in range(depth):
            self.layers.append(nn.ModuleList([get_latent_attn(**cache_args), get_latent_ff(**cache_args)]))

        self.decoder_cross_attn = PreNorm(
            queries_dim,
            Attention(queries_dim, latent_dim, heads=cross_heads, dim_head=cross_dim_head),
            context_dim=latent_dim,
        )
        self.decoder_ff = PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None
        self.to_logits = nn.Linear(queries_dim, logits_dim) if exists(logits_dim) else nn.Identity()

    def forward(self, data, mask=None, queries=None):
        b, device = data.shape[0], data.device
        x = repeat(self.latents, "n d -> b n d", b=b)
        cross_attn, cross_ff = self.cross_attend_blocks

        if self.training and self.seq_dropout_prob > 0.0:
            data, mask = dropout_seq(data, mask, self.seq_dropout_prob)

        x = cross_attn(x, context=data, mask=mask) + x
        x = cross_ff(x) + x

        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x

        if not exists(queries):
            return x

        if queries.ndim == 2:
            queries = repeat(queries, "n d -> b n d", b=b)

        latents = self.decoder_cross_attn(queries, context=x)

        if exists(self.decoder_ff):
            latents = latents + self.decoder_ff(latents)

        return self.to_logits(latents)


class EncodeBlock(nn.Module):
    def __init__(self, n_embd, n_head, n_agent):
        super().__init__()
        head_dim = max(1, n_embd // max(1, n_head))
        self.perceiver = PerceiverIO(
            depth=n_head,
            dim=n_embd,
            queries_dim=n_embd,
            num_latents=n_agent,
            latent_dim=n_embd,
            cross_heads=1,
            latent_heads=n_head,
            cross_dim_head=head_dim,
            latent_dim_head=head_dim,
            decoder_ff=True,
        )

    def forward(self, x, obs_encoding):
        perceiver_out = self.perceiver(x, queries=obs_encoding)
        return x + perceiver_out


class Encoder(nn.Module):

    def __init__(self, obs_dim, n_block, n_embd, n_head, n_agent):
        super(Encoder, self).__init__()

        self.obs_dim = obs_dim
        self.n_embd = n_embd
        self.n_agent = n_agent

        self.obs_encoder = nn.Sequential(
            init_(nn.Linear(obs_dim, n_embd), activate=True),
            nn.GELU(),
        )

        self.blocks = nn.ModuleList([EncodeBlock(n_embd, n_head, n_agent) for _ in range(n_block)])
        self.v_head = nn.Sequential(
            init_(nn.Linear(n_embd, n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(n_embd, 1))
        )

    def forward(self, obs):
        obs_embeddings = self.obs_encoder(obs) # (batch, n_agent, obs_dim)

        x = obs_embeddings
        for block in self.blocks:
            x = block(x, obs_embeddings)

        v_loc = self.v_head(x)

        return v_loc



class MAPCritic(nn.Module):
    def __init__(self, scheme, args):
        super(MAPCritic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.is_image = False

        self.input_shape = self._get_input_shape(scheme)

        self.encoder = Encoder(self.input_shape, args.n_block, args.n_embd, args.n_head, self.n_agents)

    def forward(self,  batch, t=None):
        inputs, bs, max_t = self._build_inputs(batch, t=t)
        inputs = inputs.reshape(-1, self.n_agents, self.input_shape)

        v_loc = self.encoder(inputs)

        return v_loc

    def _build_inputs(self, batch, t=None):
        bs = batch["batch_size"]
        max_t = batch["max_seq_length"] if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        inputs = []
        # observations
        inputs.append(batch["obs"][:, ts])

        # observation
        assert not (self.is_image is True and self.args.obs_individual_obs is True), \
            "In case of state image, obs_individual_obs is not supported."
        if self.args.obs_individual_obs:
            inputs.append(batch["obs"][:, ts])

        # actions (masked out by agent)
        actions = batch["actions_onehot"][:, ts].view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
        agent_mask = (1 - th.eye(self.n_agents, device=batch["device"]))
        agent_mask = agent_mask.view(-1, 1).repeat(1, self.n_actions).view(self.n_agents, -1)
        inputs.append(actions * agent_mask.unsqueeze(0).unsqueeze(0))

        # last actions
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, 0:1]).view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1))
            elif isinstance(t, int):
                inputs.append(batch["actions_onehot"][:, slice(t-1, t)].view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1))
            else:
                last_actions = th.cat([th.zeros_like(batch["actions_onehot"][:, 0:1]),
                                       batch["actions_onehot"][:, :-1]],
                                      dim=1)
                last_actions = last_actions.view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
                inputs.append(last_actions)

        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch["device"]).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))

        if self.is_image is False:
            inputs = th.cat([x.reshape(bs, max_t, self.n_agents, -1) for x in inputs], dim=-1)
        else:
            inputs[1] = th.cat([x.reshape(bs, max_t, self.n_agents, -1) for x in inputs[1:]], dim=-1)
            del inputs[2:]
            assert len(inputs) == 2, "length of inputs: {}".format(len(inputs))
        return inputs, bs, max_t

    def _get_input_shape(self, scheme):
        # state
        input_shape = scheme["obs"]["vshape"]
        if isinstance(input_shape, int):
            # observation
            if self.args.obs_individual_obs:
                input_shape += scheme["obs"]["vshape"]
            # actions
            input_shape += scheme["actions_onehot"]["vshape"][0] * self.n_agents
            # last action
            if self.args.obs_last_action:
                input_shape += scheme["actions_onehot"]["vshape"][0] * self.n_agents
            # agent id
            if self.args.obs_agent_id:
                input_shape += self.n_agents
        elif isinstance(input_shape, tuple):
            assert self.args.obs_individual_obs is False, "In case of state image, obs_individual_obs is not supported."
            self.is_image = True
            input_shape = [input_shape, 0]
            input_shape[1] += scheme["actions_onehot"]["vshape"][0] * self.n_agents
            if self.args.obs_last_action:
                input_shape[1] += scheme["actions_onehot"]["vshape"][0] * self.n_agents
            if self.args.obs_agent_id:
                input_shape[1] += self.n_agents
            input_shape = tuple(input_shape)
        return input_shape

