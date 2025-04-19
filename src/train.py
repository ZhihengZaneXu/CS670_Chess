"""
Training module for Chess RL Agent with detailed metrics tracking.
"""

import datetime
import json
import os
import time
from collections import deque

import chess
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# To this (assuming the modules are in the same directory):
from environment import create_environment
from expert import ChessExpert
from utils import create_chess_agent, selections_to_move, train_chess_agent


class ChessTrainer:
    """Class for training and tracking metrics for the Chess RL agent."""

    def __init__(
        self,
        stockfish_path=None,
        gamma=0.99,
        learning_rate=0.0003,
        tensorboard_log="./logs/",
        output_dir=f"./trained_models/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}/",
        window_size=100,  # For moving averages
    ):
        """Initialize the trainer."""
        self.tensorboard_log = tensorboard_log
        self.output_dir = output_dir
        self.window_size = window_size

        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)

        # Initialize expert
        self.expert = ChessExpert(stockfish_path) if stockfish_path else None

        # Create environment with action masking
        self.env = create_environment(
            use_expert=True if self.expert else False, include_masks=True
        )

        # Create agent
        self.agent = create_chess_agent(
            self.env,
            expert_model=self.expert,
            gamma=gamma,
            learning_rate=learning_rate,
            tensorboard_log=tensorboard_log,
        )

        # Initialize metrics tracking
        self.metrics = {
            "episode_rewards": [],
            "episode_lengths": [],
            "perfect_move_accuracy": [],
            "piece_selection_accuracy": [],
            "destination_accuracy": [],
            "win_rate": [],
            "draw_rate": [],
            "loss_rate": [],
            "avg_rewards": [],
            "avg_episode_length": [],
            "avg_perfect_accuracy": [],
            "avg_piece_accuracy": [],
            "avg_dest_accuracy": [],
        }

        # For detailed move-by-move tracking
        self.move_stats = []

        # For running averages
        self.recent_rewards = deque(maxlen=window_size)
        self.recent_lengths = deque(maxlen=window_size)
        self.recent_perfect = deque(maxlen=window_size)
        self.recent_piece = deque(maxlen=window_size)
        self.recent_dest = deque(maxlen=window_size)

        # Game outcome tracking
        self.wins = 0
        self.draws = 0
        self.losses = 0
        self.total_games = 0

    def load_pretrained(self, model_path):
        """Load a pretrained model."""
        self.agent = self.agent.load(model_path, env=self.env)
        print(f"Loaded pretrained model from {model_path}")

    def _evaluate_move_accuracy(self, chosen_move, expert_move):
        """
        Evaluate the accuracy of a move compared to the expert's move.

        Returns:
            tuple: (perfect_match, piece_match, destination_match)
        """
        if chosen_move is None or expert_move is None:
            return False, False, False

        perfect_match = chosen_move == expert_move
        piece_match = chosen_move.from_square == expert_move.from_square
        dest_match = chosen_move.to_square == expert_move.to_square

        return perfect_match, piece_match, dest_match

    def _get_running_averages(self):
        """Calculate running averages of metrics."""
        avg_reward = np.mean(self.recent_rewards) if self.recent_rewards else 0
        avg_length = np.mean(self.recent_lengths) if self.recent_lengths else 0
        avg_perfect = np.mean(self.recent_perfect) if self.recent_perfect else 0
        avg_piece = np.mean(self.recent_piece) if self.recent_piece else 0
        avg_dest = np.mean(self.recent_dest) if self.recent_dest else 0

        return avg_reward, avg_length, avg_perfect, avg_piece, avg_dest

    def train(
        self, total_episodes=1000, max_steps=100, eval_interval=10, save_interval=100
    ):
        """
        Train the agent and track metrics.

        Args:
            total_episodes: Total number of episodes to train for
            max_steps: Maximum steps per episode
            eval_interval: Episodes between evaluations
            save_interval: Episodes between model checkpoints
        """
        # Start timing
        start_time = time.time()

        from utils import FrequentLoggingCallback

        frequent_logging = FrequentLoggingCallback(log_freq=10)
        total_timesteps = total_episodes * max_steps

        self.agent.learn(
            total_timesteps=total_timesteps,
            callback=frequent_logging,
            tb_log_name="chess_training",
        )

        # Main training loop
        for episode in tqdm(range(1, total_episodes + 1)):
            # Reset environment
            obs = self.env.reset()
            done = False
            episode_reward = 0
            steps = 0

            # Track move accuracy for this episode
            episode_perfect_matches = 0
            episode_piece_matches = 0
            episode_dest_matches = 0
            episode_move_stats = []

            # Episode loop
            while not done and steps < max_steps:
                # Get action from agent (during training, agent.predict handles exploration)
                action, _ = self.agent.predict(obs, deterministic=False)

                # Convert action to chess move
                piece_selection, move_selection = action
                valid_moves = list(self.env.chess_env.board.legal_moves)

                chess_move = selections_to_move(
                    piece_selection, move_selection, valid_moves
                )

                # Get expert move for comparison
                expert_move = None
                if self.expert:
                    expert_move = self.expert.get_best_move(self.env.chess_env.board)

                # Evaluate move accuracy
                if chess_move and expert_move:
                    perfect, piece, dest = self._evaluate_move_accuracy(
                        chess_move, expert_move
                    )
                    episode_perfect_matches += int(perfect)
                    episode_piece_matches += int(piece)
                    episode_dest_matches += int(dest)

                    # Store detailed move stats
                    move_info = {
                        "episode": episode,
                        "step": steps,
                        "board_fen": self.env.chess_env.board.fen(),
                        "chosen_move": chess_move.uci(),
                        "expert_move": expert_move.uci() if expert_move else None,
                        "perfect_match": perfect,
                        "piece_match": piece,
                        "dest_match": dest,
                    }
                    episode_move_stats.append(move_info)

                # Execute step in environment
                obs, reward, done, info = self.env.step(action)
                episode_reward += reward
                steps += 1

            # End of episode
            self.total_games += 1

            # Record game outcome
            if "result" in info:
                if info["result"] == "1-0":
                    self.wins += 1
                elif info["result"] == "0-1":
                    self.losses += 1
                else:  # Draw
                    self.draws += 1

            # Calculate move accuracies as percentages
            if steps > 0:
                perfect_accuracy = episode_perfect_matches / steps
                piece_accuracy = episode_piece_matches / steps
                dest_accuracy = episode_dest_matches / steps
            else:
                perfect_accuracy = piece_accuracy = dest_accuracy = 0

            # Update metrics
            self.metrics["episode_rewards"].append(episode_reward)
            self.metrics["episode_lengths"].append(steps)
            self.metrics["perfect_move_accuracy"].append(perfect_accuracy)
            self.metrics["piece_selection_accuracy"].append(piece_accuracy)
            self.metrics["destination_accuracy"].append(dest_accuracy)

            # Update running averages
            self.recent_rewards.append(episode_reward)
            self.recent_lengths.append(steps)
            self.recent_perfect.append(perfect_accuracy)
            self.recent_piece.append(piece_accuracy)
            self.recent_dest.append(dest_accuracy)

            avg_reward, avg_length, avg_perfect, avg_piece, avg_dest = (
                self._get_running_averages()
            )

            self.metrics["avg_rewards"].append(avg_reward)
            self.metrics["avg_episode_length"].append(avg_length)
            self.metrics["avg_perfect_accuracy"].append(avg_perfect)
            self.metrics["avg_piece_accuracy"].append(avg_piece)
            self.metrics["avg_dest_accuracy"].append(avg_dest)

            # Calculate win/draw/loss rates
            if self.total_games > 0:
                win_rate = self.wins / self.total_games
                draw_rate = self.draws / self.total_games
                loss_rate = self.losses / self.total_games
            else:
                win_rate = draw_rate = loss_rate = 0

            self.metrics["win_rate"].append(win_rate)
            self.metrics["draw_rate"].append(draw_rate)
            self.metrics["loss_rate"].append(loss_rate)

            # Store detailed move stats
            self.move_stats.extend(episode_move_stats)

            # Print metrics at intervals
            if episode % eval_interval == 0:
                elapsed_time = time.time() - start_time
                print(
                    f"\nEpisode {episode}/{total_episodes} | Time: {elapsed_time:.1f}s"
                )
                print(f"Reward: {episode_reward:.2f} | Avg Reward: {avg_reward:.2f}")
                print(f"Length: {steps} | Avg Length: {avg_length:.1f}")
                print(
                    f"Perfect Accuracy: {perfect_accuracy:.2%} | Avg: {avg_perfect:.2%}"
                )
                print(f"Piece Accuracy: {piece_accuracy:.2%} | Avg: {avg_piece:.2%}")
                print(
                    f"Destination Accuracy: {dest_accuracy:.2%} | Avg: {avg_dest:.2%}"
                )
                print(
                    f"Win Rate: {win_rate:.2%} | Draw Rate: {draw_rate:.2%} | Loss Rate: {loss_rate:.2%}"
                )

                # Save metrics at intervals
                self._save_metrics()

            # Save model checkpoint at intervals
            if episode % save_interval == 0:
                checkpoint_path = os.path.join(
                    self.output_dir, "checkpoints", f"model_ep{episode}"
                )
                self.agent.save(checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")

        # Save final model and metrics
        final_model_path = os.path.join(self.output_dir, "final_model")
        self.agent.save(final_model_path)
        self._save_metrics()

        # Generate plots
        self._generate_plots()

        # Final stats
        elapsed_time = time.time() - start_time
        print(f"\nTraining completed in {elapsed_time:.1f} seconds")
        print(f"Final model saved to {final_model_path}")

        return self.agent

    def _save_metrics(self):
        """Save metrics to JSON file."""
        metrics_path = os.path.join(self.output_dir, "metrics", "training_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=2)

        # Save detailed move stats
        moves_path = os.path.join(self.output_dir, "metrics", "move_stats.json")
        with open(moves_path, "w") as f:
            json.dump(self.move_stats, f, indent=2)

    def _generate_plots(self):
        """Generate and save plots of training metrics."""
        plots_dir = os.path.join(self.output_dir, "metrics", "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # Plot rewards
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics["episode_rewards"], alpha=0.3, label="Episode Reward")
        plt.plot(self.metrics["avg_rewards"], label="Avg Reward")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Training Rewards")
        plt.legend()
        plt.savefig(os.path.join(plots_dir, "rewards.png"))

        # Plot episode lengths
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics["episode_lengths"], alpha=0.3, label="Episode Length")
        plt.plot(self.metrics["avg_episode_length"], label="Avg Length")
        plt.xlabel("Episode")
        plt.ylabel("Steps")
        plt.title("Episode Lengths")
        plt.legend()
        plt.savefig(os.path.join(plots_dir, "episode_lengths.png"))

        # Plot move accuracies
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics["avg_perfect_accuracy"], label="Perfect Move")
        plt.plot(self.metrics["avg_piece_accuracy"], label="Piece Selection")
        plt.plot(self.metrics["avg_dest_accuracy"], label="Destination")
        plt.xlabel("Episode")
        plt.ylabel("Accuracy")
        plt.title("Move Accuracy")
        plt.legend()
        plt.savefig(os.path.join(plots_dir, "move_accuracy.png"))

        # Plot game outcomes
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics["win_rate"], label="Win Rate")
        plt.plot(self.metrics["draw_rate"], label="Draw Rate")
        plt.plot(self.metrics["loss_rate"], label="Loss Rate")
        plt.xlabel("Episode")
        plt.ylabel("Rate")
        plt.title("Game Outcomes")
        plt.legend()
        plt.savefig(os.path.join(plots_dir, "game_outcomes.png"))


def train_chess_model(args):
    """Main training function."""
    # Create trainer
    trainer = ChessTrainer(
        stockfish_path=args.stockfish_path,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        tensorboard_log=args.tensorboard_log,
        output_dir=args.output_dir,
    )

    # Load pretrained model if specified
    if args.load_pretrained:
        trainer.load_pretrained(args.load_pretrained)

    # Train the model
    trainer.train(
        total_episodes=args.episodes,
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
    )


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Train a chess RL agent")
    parser.add_argument(
        "--stockfish_path",
        type=str,
        default="/home/pd468/cs670/project/expert_models/stockfish/stockfish-ubuntu-x86-64-avx2",
        help="Path to Stockfish executable",
    )
    parser.add_argument(
        "--episodes", type=int, default=1000, help="Number of training episodes"
    )
    parser.add_argument(
        "--max_steps", type=int, default=100, help="Maximum steps per episode"
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument(
        "--learning_rate", type=float, default=0.0003, help="Learning rate"
    )
    parser.add_argument(
        "--eval_interval", type=int, default=10, help="Episodes between evaluations"
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=100,
        help="Episodes between saving checkpoints",
    )
    parser.add_argument(
        "--tensorboard_log",
        type=str,
        default="./logs/",
        help="Directory for TensorBoard logs",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./trained_models/",
        help="Directory for saving models",
    )
    parser.add_argument(
        "--load_pretrained", type=str, default=None, help="Path to pretrained model"
    )

    args = parser.parse_args()
    train_chess_model(args)
