import argparse
from utils.config import load_config
from utils.logging import capture_all_output, log_config_details

def main():
    parser = argparse.ArgumentParser(description="Train RL agent.")
    parser.add_argument("--config", type=str, default="Pong-v5_objects_ppo", help="Config ID (default: Pong-v5_ram_ppo)")
    parser.add_argument("--algo", type=str, default=None, help="Agent type (optional, extracted from config if not provided)")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory for log files (default: logs)")
    args = parser.parse_args()

    config = load_config(args.config, args.algo)
    
    # Override resume flag if provided via command line
    if args.resume:
        config.resume = True
    
    # Set up run manager first to get run-specific directory
    from dataclasses import asdict
    from pytorch_lightning.loggers import WandbLogger
    from utils.run_manager import RunManager

    # Use regular WandbLogger to get run ID
    project_name = config.env_id.replace("/", "-").replace("\\", "-")
    experiment_name = f"{config.algo_id}-{config.seed}"
    wandb_logger = WandbLogger(
        project=project_name,
        name=experiment_name,
        log_model=True,
        config=asdict(config)
    )
    
    # Setup run directory management to get run-specific logs directory
    run_manager = RunManager()
    run_dir = run_manager.setup_run_directory(wandb_logger.experiment)
    run_logs_dir = str(run_manager.get_logs_dir())
    
    print(f"Run directory: {run_dir}")
    print(f"Run ID: {run_manager.run_id}")
    print(f"Logs will be saved to: {run_logs_dir}")
    
    # Set up comprehensive logging using run-specific logs directory
    with capture_all_output(config=config, log_dir=run_logs_dir):
        print(f"=== Training Session Started ===")
        # Build command string, only include --algo if it was explicitly provided
        cmd_parts = ['python', 'train.py', '--config', args.config]
        if args.algo is not None:
            cmd_parts.extend(['--algo', args.algo])
        if args.resume:
            cmd_parts.append('--resume')
        print(f"Command: {' '.join(cmd_parts)}")
        
        # Log configuration details
        log_config_details(config)
        
        from stable_baselines3.common.utils import set_random_seed
        set_random_seed(config.seed)

        from agents import create_agent
        agent = create_agent(config)
        
        # Pass the wandb logger and run manager to the agent
        agent.wandb_logger = wandb_logger
        agent.run_manager = run_manager
        
        # Save configuration to run directory
        config_path = run_manager.save_config(config)
        print(f"Configuration saved to: {config_path}")
        
        print(str(agent))

        print("Starting training...")
        agent._run_training()
        print("Training completed.")
        
if __name__ == "__main__":
    main()
