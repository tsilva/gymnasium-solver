from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecVideoRecorder
from utils.rollouts import RolloutCollector
from utils.models import ActorCritic

env_id = "CartPole-v1"
N = 4
video_folder = "videos"

policy_model = ActorCritic(4, 2)
env = make_vec_env(
    env_id,
    n_envs=N,
    env_kwargs={"render_mode": "rgb_array"}
)
env = VecVideoRecorder(
    env,
    video_folder=video_folder,
    record_video_trigger=lambda step: step == 0,   # start immediately
    #video_length=video_length,
    #name_prefix=f"{env_id}-{N}x-grid",
)
collector = RolloutCollector(env, policy_model, n_steps=500)
collector.collect()
print(collector.get_ep_rew_mean())