
def main():
    from utils.training import create_trainer
    from utils.rollouts import SyncRolloutCollector # TODO: restore async functionality
    from utils.models import PolicyNet, ValueNet
    from learners.ppo import PPOLearner
    from learners.reinforce import REINFORCELearner
    from utils.config import load_config
    from tsilva_notebook_utils.gymnasium import log_env_info
    from utils.environment import setup_environment
    from utils.rollouts import SyncRolloutCollector
    from utils.models import PolicyNet, ValueNet

    ENV_ID = "CartPole-v1"
    #ENV_ID = "LunarLander-v3"
    #ALGORITHM = "reinforce"
    ALGO_ID = "reinforce"


    # Load secrets:


    from tsilva_notebook_utils.colab import load_secrets_into_env
    _ = load_secrets_into_env(['WANDB_API_KEY'])


    # Load training configuration:


    CONFIG = load_config(ENV_ID, ALGO_ID)
    print(f"Loaded config for {ENV_ID} with {ALGO_ID} algorithm:")
    print(CONFIG)


    # Build environment:


    build_env_fn = setup_environment(CONFIG, n_envs=1) # TODO: consider getting rid of this method or moving everything inside it
    env = build_env_fn(CONFIG.seed)
    log_env_info(env)



    # TODO: make rollout collector clone models?
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    policy_model = PolicyNet(input_dim, output_dim, CONFIG.hidden_dims)
    value_model = ValueNet(input_dim, CONFIG.hidden_dims) if ALGO_ID == "ppo" else None # TODO: softcode this better

    train_env = build_env_fn(CONFIG.seed)
    train_rollout_collector = SyncRolloutCollector(
        train_env,
        policy_model,
        value_model=value_model,
        n_steps=CONFIG.train_rollout_steps
    )

    eval_env = build_env_fn(CONFIG.seed + 1000)
    eval_rollout_collector = SyncRolloutCollector(
        # TODO: pass env factory and rebuild env on start/stop? this allows using same rollout collector for final evaluation
        eval_env,  # Use a different seed for evaluation
        policy_model,
        n_episodes=8,
        deterministic=True
    )

    algo_id = ALGO_ID.lower()
    if algo_id == "ppo": agent = PPOLearner(CONFIG, train_rollout_collector, policy_model, value_model, eval_rollout_collector=eval_rollout_collector)
    elif algo_id == "reinforce": agent = REINFORCELearner(CONFIG, train_rollout_collector, policy_model, eval_rollout_collector=eval_rollout_collector)

    # Create trainer with W&B logging
    # TODO: infer most args
    trainer = create_trainer(CONFIG, project_name=ENV_ID, run_name=f"{ALGO_ID}-{CONFIG.seed}")

    # Fit the model
    trainer.fit(agent)

    #from evaluation import render_rollouts
    #render_rollouts(eval_rollout_collector, n_episodes=16)

if __name__ == "__main__":
    main()


