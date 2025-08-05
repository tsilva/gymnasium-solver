from .env_wrapper_registry import EnvWrapperRegistry
from .discrete_to_onehot import DiscreteToOneHot
from .PongV5.feature_extractor import PongV5_FeatureExtractor
from .MountainCarV0.reward_shaper import MountainCarV0_RewardShaper

EnvWrapperRegistry.register([
    DiscreteToOneHot,
    PongV5_FeatureExtractor,
    MountainCarV0_RewardShaper
])