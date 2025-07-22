# %% [markdown]
# # Environment Solver

# %% [markdown]
# Make notebook autoreload when imported repo modules change:

# %%

# %% [markdown]
# Define which environment we want to solve and with which algorithm:

# %%
ENV_ID = "CartPole-v1"
#ENV_ID = "LunarLander-v3"
#ALGORITHM = "reinforce"
ALGO_ID = "ppo"

# %% [markdown]
# Load secrets:

# %%
from tsilva_notebook_utils.colab import load_secrets_into_env
_ = load_secrets_into_env(['WANDB_API_KEY'])

# %% [markdown]
# Load training configuration:

# %%
from utils.config import load_config
CONFIG = load_config(ENV_ID, ALGO_ID)
print(f"Loaded config for {ENV_ID} with {ALGO_ID} algorithm:")
print(CONFIG)

# %% [markdown]
# Build environment:

# %%
from tsilva_notebook_utils.gymnasium import log_env_info
from utils.environment import setup_environment
build_env_fn = setup_environment(CONFIG) # TODO: consider getting rid of this method or moving everything inside it
env = build_env_fn(CONFIG.seed)
log_env_info(env)

# %%
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

# %%
#trajectories, stats = train_rollout_collector.collect_rollouts() # TODO: consider making get rollout be async method even in sync rollout?
#sum(1 if t else 0 for t in trajectories[3]), len(trajectories[3])

# %% [markdown]
# Define models:

# %%
from utils.training import create_trainer
from utils.rollouts import SyncRolloutCollector # TODO: restore async functionality
from utils.models import PolicyNet, ValueNet
from learners.ppo import PPOLearner
from learners.reinforce import REINFORCELearner

# TODO: make rollout collector clone models?
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
policy_model = PolicyNet(input_dim, output_dim, CONFIG.hidden_dims)
value_model = ValueNet(input_dim, CONFIG.hidden_dims) if ALGO_ID == "ppo" else None # TODO: softcode this better

train_rollout_collector = SyncRolloutCollector(
    build_env_fn(CONFIG.seed),
    policy_model,
    value_model=value_model,
    n_steps=CONFIG.train_rollout_steps
)
eval_rollout_collector = SyncRolloutCollector(
    # TODO: pass env factory and rebuild env on start/stop? this allows using same rollout collector for final evaluation
    build_env_fn(CONFIG.seed + 1000),  # Use a different seed for evaluation
    policy_model,
    n_steps=CONFIG.train_rollout_steps # TODO: change this
)
algo_id = ALGO_ID.lower()
if algo_id == "ppo": agent = PPOLearner(CONFIG, train_rollout_collector, policy_model, value_model, eval_rollout_collector=eval_rollout_collector)
elif algo_id == "reinforce": agent = REINFORCELearner(CONFIG, train_rollout_collector, policy_model, eval_rollout_collector=eval_rollout_collector)

# Create trainer with W&B logging
# TODO: infer most args
trainer = create_trainer(CONFIG, project_name=ENV_ID, run_name=f"{ALGO_ID}-{CONFIG.seed}")

# Fit the model
trainer.fit(agent)

# %%
from utils.evaluation import evaluate_agent

# Evaluate agent and render episodes
# TODO: evaluate agent should receive rollout collector as an argument, not the agent and build_env_fn
results = evaluate_agent(
    agent, 
    build_env_fn, 
    n_episodes=8, 
    deterministic=True, 
    render=True,
    grid=(2, 2), 
    text_color=(0, 0, 0), 
    out_dir="./tmp"
)

print(f"Mean reward: {results['mean_reward']:.2f}")


