## Setup

### Conda 

Create a conda environment with python 3.11

```bash
conda create -n map python=3.11
conda activate map
```

Install project (see common installation issues for pymarlzooplus [here](https://github.com/AILabDsUnipi/pymarlzooplus/tree/main?tab=readme-ov-file#known-issues))

```bash
pip install -e .
```
<!-- 
### Docker

Build:

```bash
docker build -t map:latest .
```

Launch a training run (Set the warehouse layout with `env_args.key` and other params as necessary. See `config` directory for all options.)

```bash
docker run --rm -it \
    -v "$PWD/pymarlzooplus/results:/app/pymarlzooplus/results" \
    map:latest \
    --config=map_dec \
    --env-config=gymma \
    with \
    env_args.time_limit=500 \
    env_args.key="rware:rware-tiny-4ag-hard-v1" \
    env_args.seed=742

# env_args.key options:
# [rware:rware-small-4ag-hard-v1, rware:rware-tiny-4ag-hard-v1, rware:rware-tiny-2ag-hard-v1]
```

Evaluate on an existing checkpoint

```bash
docker run --rm -it \
    map:latest \
    --config=map_dec \
    --env-config=gymma \
    with \
    env_args.key="rware:rware-tiny-4ag-hard-v1" \
    evaluate=True \
    checkpoint_path="pymarlzooplus/results/sacred/map_dec/rware:rware-tiny-4ag-hard-v1/4" \
    load_step=100000 \
    test_nepisode=100
```

### Rendering

To see a trained policy in action, the RWARE renderer is a native pyglet/OpenGL window. Here are 3 ways to run it on a saved checkpoint:

#### 1) Linux (Recommended): run on the host (assumes conda setup from above)

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

#### 2) macOS (requires XQuartz)

- Install and start **XQuartz**
- In XQuartz settings, enable **“Allow connections from network clients”**
- Then run:

```bash
xhost + 127.0.0.1
docker run --rm -it \
    -e DISPLAY=host.docker.internal:0 \
    -v "$PWD/pymarlzooplus/results:/app/pymarlzooplus/results" \
    map:latest \
    python pymarlzooplus/main.py \
        --config=map_dec \
        --env-config=gymma \
        with \
        env_args.key="rware:rware-tiny-4ag-hard-v1" \
        env_args.time_limit=500 \
        checkpoint_path="pymarlzooplus/results/sacred/map_dec/rware:rware-tiny-4ag-hard-v1/0/models" \
        load_step = 0 \
        evaluate=True render=True render_sleep_time=0.4
```

#### 3) Windows (requires VcXsrv / X server)

- Install and start **VcXsrv** (or Xming)
- Run it with **"Disable access control"** (quick/easy), or configure proper access control
- Then run (PowerShell example):

```powershell
docker run --rm -it \
    -e DISPLAY=host.docker.internal:0 \
    -v "${PWD}\pymarlzooplus\results:/app/pymarlzooplus/results" \
    map:latest \
    python pymarlzooplus/main.py \
        --config=map_dec \
        --env-config=gymma \
        with \
        env_args.key="rware:rware-tiny-4ag-hard-v1" \
        env_args.time_limit=500 \
        checkpoint_path="pymarlzooplus/results/sacred/map_dec/rware:rware-tiny-4ag-hard-v1/0/models" \
        load_step = 0 \
        evaluate=True render=True render_sleep_time=0.4
``` -->