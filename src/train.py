import argparse
import datetime
import os

import chess
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)

from environment import create_environment
from utils import FrequentLoggingCallback, create_chess_agent


class CustomMetricsCallback(BaseCallback):
    """Log custom game outcome metrics after each episode."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.wins = 0
        self.draws = 0
        self.losses = 0
        self.total_games = 0

    def _on_step(self) -> bool:
        # 'infos' is a list of info dicts for each env in the batch
        infos = self.locals.get("infos")
        dones = self.locals.get("dones")
        if infos is None or dones is None:
            return True
        for info, done in zip(infos, dones):
            if done and "result" in info:
                result = info["result"]
                self.total_games += 1
                if result == "1-0":
                    if self.training_env.envs[0].agent_color == chess.WHITE:
                        self.wins += 1
                    else:
                        self.losses += 1
                elif result == "0-1":
                    if self.training_env.envs[0].agent_color == chess.BLACK:
                        self.wins += 1
                    else:
                        self.losses += 1
                else:
                    self.draws += 1
                # Record rates
                self.logger.record("custom/win_rate", self.wins / self.total_games)
                self.logger.record("custom/draw_rate", self.draws / self.total_games)
                self.logger.record("custom/loss_rate", self.losses / self.total_games)
        return True


def main():
    parser = argparse.ArgumentParser(description="Train a hierarchical chess RL agent")
    parser.add_argument(
        "--episodes", type=int, default=1000, help="Number of episodes to train"
    )
    parser.add_argument(
        "--max_steps", type=int, default=100, help="Max steps per episode"
    )
    parser.add_argument(
        "--stockfish_path",
        type=str,
        default="/home/pd468/cs670/project/expert_models/stockfish/stockfish-ubuntu-x86-64-avx2",
        help="Enable expert guidance by specifying Stockfish path",
    )
    parser.add_argument(
        "--stockfish_depth", type=int, default=1, help="Stockfish search depth"
    )
    parser.add_argument("--n_steps", type=int, default=512, help="Steps per PPO update")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument(
        "--learning_rate", type=float, default=0.0003, help="PPO learning rate"
    )
    parser.add_argument(
        "--tensorboard_log",
        type=str,
        default=None,
        help="Directory for TensorBoard logs (optional)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(
            "./trained_models", datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        ),
        help="Directory to save checkpoints and final model",
    )
    parser.add_argument(
        "--agent_color",
        choices=["white", "black"],
        default="white",
        help="Agent playing color",
    )
    parser.add_argument(
        "--load_pretrained",
        type=str,
        default=None,
        help="Path to pretrained model (optional)",
    )
    args = parser.parse_args()

    if args.tensorboard_log is None:
        args.tensorboard_log = os.path.join(args.output_dir, "tensorboard")
    os.makedirs(args.tensorboard_log, exist_ok=True)

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create environment and agent
    agent_color = chess.WHITE if args.agent_color == "white" else chess.BLACK
    env = create_environment(
        use_expert=bool(args.stockfish_path),
        include_masks=True,
        stockfish_opponent=None,
        stockfish_depth=args.stockfish_depth,
        agent_color=agent_color,
    )
    agent = create_chess_agent(
        env,
        gamma=args.gamma,
        n_steps=args.n_steps,
        learning_rate=args.learning_rate,
        tensorboard_log=args.tensorboard_log,
    )

    # Optionally load a pretrained model
    if args.load_pretrained:
        agent = agent.load(args.load_pretrained, env=env)
        print(f"Loaded pretrained model from {args.load_pretrained}")

    # Define callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=1000, save_path=checkpoint_dir, name_prefix="model"
    )
    frequent_cb = FrequentLoggingCallback(log_freq=10)
    custom_cb = CustomMetricsCallback(verbose=1)

    # Train and save checkpoints
    total_timesteps = args.episodes * args.max_steps
    callbacks = CallbackList([checkpoint_cb, frequent_cb, custom_cb])
    agent.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        tb_log_name="chess_run",
    )

    # Save final model
    final_path = os.path.join(args.output_dir, "final_model")
    agent.save(final_path)
    print(f"Training complete. Model saved to {final_path}")


if __name__ == "__main__":
    main()
