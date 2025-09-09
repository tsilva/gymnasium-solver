def build_agent(config, *args, **kwargs):
    algo_id = config.algo_id
    if algo_id == "ppo": 
        from .ppo import PPO
        return PPO(config, *args, **kwargs)
    elif algo_id == "reinforce": 
        from .reinforce import REINFORCE
        return REINFORCE(config, *args, **kwargs)