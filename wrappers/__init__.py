from .env_wrapper_registry import EnvWrapperRegistry
from .reward_shaper_mountaincar_v0 import RewardShaper_MountainCarV0
from .discrete_to_onehot import DiscreteToOneHot

EnvWrapperRegistry.register([
    RewardShaper_MountainCarV0,
    DiscreteToOneHot
])