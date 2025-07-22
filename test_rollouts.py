ENV_ID = "CartPole-v1"
#ENV_ID = "LunarLander-v3"
#ALGORITHM = "reinforce"
ALGO_ID = "ppo"

from utils.config import load_config
CONFIG = load_config(ENV_ID, ALGO_ID)
print(f"Loaded config for {ENV_ID} with {ALGO_ID} algorithm:")
print(CONFIG)

from tsilva_notebook_utils.gymnasium import log_env_info
from utils.environment import setup_environment
build_env_fn = setup_environment(CONFIG) # TODO: consider getting rid of this method or moving everything inside it
env = build_env_fn(CONFIG.seed)
log_env_info(env)

from utils.rollouts import SyncRolloutCollector
from utils.models import PolicyNet, ValueNet
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n if hasattr(env.action_space, 'n') else env.action_space.shape[0]
policy_model = PolicyNet(obs_dim, act_dim, CONFIG.hidden_dims)
value_model = ValueNet(obs_dim, CONFIG.hidden_dims) if ALGO_ID == "ppo" else None
train_rollout_collector = SyncRolloutCollector(
    build_env_fn(CONFIG.seed),
    policy_model,
    value_model=value_model,
    n_steps=CONFIG.train_rollout_steps
)

while True:
    trajectories = train_rollout_collector.collect_rollouts()
    dones = sum(1 if t else 0 for t in trajectories[3]), len(trajectories[3])
    print(f"Collected {len(trajectories[0])} trajectories with {dones[0]} finished episodes.")
