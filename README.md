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