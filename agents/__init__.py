def build_agent(config, *args, **kwargs):
    algo_id = config.algo_id
    if algo_id == "ppo": 
        from .ppo import PPOAgent
        return PPOAgent(config, *args, **kwargs)
    elif algo_id == "p3o":
        from .p3o import P3OAgent
        return P3OAgent(config, *args, **kwargs)
    elif algo_id == "reinforce": 
        from .reinforce import REINFORCEAgent
        return REINFORCEAgent(config, *args, **kwargs)
