# Import all environment wrappers to register them
from .env_wrapper_registry import EnvWrapperRegistry
from .reward_shaper_mountaincar_v0 import RewardShaper_MountainCarV0
from .vec_video_recorder import VecVideoRecorder

EnvWrapperRegistry.register('RewardShaper_MountainCarV0', RewardShaper_MountainCarV0)