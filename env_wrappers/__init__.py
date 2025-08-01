from .env_wrapper_registry import EnvWrapperRegistry
from .reward_shaper_mountaincar_v0 import RewardShaper_MountainCarV0

EnvWrapperRegistry.register([
    RewardShaper_MountainCarV0
])