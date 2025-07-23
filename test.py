
def main():
    from agents.reinforce import REINFORCELearner
    agent = REINFORCELearner("CartPole-v1", "reinforce", n_envs="auto") # TODO: build env_fn inside?
    agent.train()
    #agent.eval()

if __name__ == "__main__":
    main()
