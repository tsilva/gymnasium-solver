
def main():
    from agents import create_agent
    agent = create_agent("reinforce", "CartPole-v1", n_envs="auto") # TODO: build env_fn inside?
    agent.train()
    #agent.eval()

if __name__ == "__main__":
    main()
