Multi-Agent Perceiver[^1][^2] Critic for Robotic Warehouse [(RWARE)](https://github.com/semitable/robotic-warehouse.git) Tasks implemented in pymarlzooplus[^3].

## Setup

See instructions for Docker [here](docker/README.md). Alternatively, create a conda environment with python 3.11

```bash
conda create -n map python=3.11
conda activate map
```

Install project (see common installation issues for pymarlzooplus [here](https://github.com/AILabDsUnipi/pymarlzooplus/tree/main?tab=readme-ov-file#known-issues))

```bash
pip install -e .
```

## Usage

### Train

Launch a training run (modify the config args in the command or modify them in the config files in the [`config`](pymarlzooplus/config/) directory)

```bash
python pymarlzooplus/main.py \
    --config=map_dec \
    --env-config=gymma \
    with \
    env_args.time_limit=500 \
    env_args.key="rware:rware-tiny-4ag-hard-v1" \
    env_args.seed=742

# env_args.key options:
# [rware:rware-small-4ag-hard-v1, rware:rware-tiny-4ag-hard-v1, rware:rware-tiny-2ag-hard-v1]
```

### Eval

Evaluate a saved checkpoint

```bash
python pymarlzooplus/main.py \
    --config=map_dec \
    --env-config=gymma \
    with \
    env_args.key="rware:rware-tiny-4ag-hard-v1" \
    env_args.time_limit=500 \
    checkpoint_path="pymarlzooplus/results/sacred/map_dec/rware:rware-tiny-4ag-hard-v1/1/models" \
    evaluate=True \
    load_step=100000 \
    test_nepisode=100

# load_step: Load model trained on this many timesteps (0 if choose max possible)
# checkpoint_path: Load model from this path
```

### Render

To see a trained policy in action, run

```bash
python pymarlzooplus/main.py \
    --config=map_dec \
    --env-config=gymma \
    with \
    env_args.key="rware:rware-tiny-4ag-hard-v1" \
    env_args.time_limit=500 \
    checkpoint_path="pymarlzooplus/results/sacred/map_dec/rware:rware-tiny-4ag-hard-v1/0/models" \
    load_step=0 \
    evaluate=True render=True render_sleep_time=0.4

# render_sleep_time: sleep time between renders (only when render == True)
# load_step: Load model trained on this many timesteps (0 if choose max possible)
# checkpoint_path: Load model from this path
```

## References

This implementation is largely adapted from the [`perceiver-pytorch`](https://github.com/lucidrains/perceiver-pytorch.git) and the [`pymarlzooplus`](https://github.com/AILabDsUnipi/pymarlzooplus.git) repos.

[^1]: Jaegle, A., Gimeno, F., Brock, A., Zisserman, A., Vinyals, O., & Carreira, J. (2021). Perceiver: General Perception with Iterative Attention. [ArXiv:2103.03206](https://arxiv.org/abs/2103.03206)

[^2]: Jaegle, A., Borgeaud, S., Alayrac, J.-B., Doersch, C., Ionescu, C., Ding, D., Koppula, S., Zoran, D., Brock, A., Shelhamer, E., Hénaff, O., Botvinick, M. M., Zisserman, A., Vinyals, O., & Carreira, J. (2022). Perceiver IO: A General Architecture for Structured Inputs & Outputs. [ArXiv:2107.14795](https://arxiv.org/abs/2107.14795)

[^3]: G. Papadopoulos, A. Kontogiannis, F. Papadopoulou, C. Poulianou, I. Koumentis, & G. Vouros. An Extended Benchmarking of Multi-Agent Reinforcement Learning Algorithms in Complex Fully Cooperative
Tasks. [arXiv:2502.04773](https://arxiv.org/abs/2502.04773)