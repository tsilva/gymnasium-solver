from .discrete_to_binary import DiscreteToBinary
from .env_wrapper_registry import EnvWrapperRegistry
from .MountainCarV0.reward_shaper import MountainCarV0_RewardShaper
from .PongV5.feature_extractor import PongV5_FeatureExtractor
from .PongV5.reward_shaper import PongV5_RewardShaper
from .CartPoleV1.reward_shaper import CartPoleV1_RewardShaper

EnvWrapperRegistry.register([
    DiscreteToBinary,
    PongV5_FeatureExtractor,
    PongV5_RewardShaper,
    MountainCarV0_RewardShaper,
    CartPoleV1_RewardShaper
])