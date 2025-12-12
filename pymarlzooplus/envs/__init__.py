# Needed for the imports
REGISTRY_availability = [
    "gymma",
]

from functools import partial  # noqa: E402

from pymarlzooplus.envs.multiagentenv import MultiAgentEnv  # noqa: E402
from pymarlzooplus.envs.gym_wrapper import _GymmaWrapper  # noqa: E402

# Gymnasium registrations
import pymarlzooplus.envs.rware_v1_registration  # noqa: E402


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {
    "gymma": partial(env_fn, env=_GymmaWrapper),
}