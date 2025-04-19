"""
Visualization module for Chess RL Agent training metrics.
This module can load saved metrics and generate plots without requiring a display.
"""

import argparse
import json
import os

import matplotlib

# Force matplotlib to use a non-interactive backend that doesn't require a display
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


class ChessMetricsVisualizer:
    """Class for visualizing training metrics for Chess RL agent."""

    def __init__(self, metrics_dir, output_dir=None):
        """
        Initialize the visualizer.

        Args:
            metrics_dir: Directory containing the training metrics JSON files
            output_dir: Directory to save the generated plots (default: metrics_dir/plots)
        """
        self.metrics_dir = metrics_dir

        # Set output directory
        if output_dir is None:
            self.output_dir = os.path.join(metrics_dir, "plots")
        else:
            self.output_dir = output_dir

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize metrics
        self.metrics = None
        self.move_stats = None

    def load_metrics(self):
        """Load metrics from JSON files."""
        # Load main metrics
        metrics_path = os.path.join(self.metrics_dir, "training_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                self.metrics = json.load(f)
            print(f"Loaded training metrics from {metrics_path}")
        else:
            print(f"Warning: Could not find training metrics at {metrics_path}")

        # Load move stats if available
        moves_path = os.path.join(self.metrics_dir, "move_stats.json")
        if os.path.exists(moves_path):
            with open(moves_path, "r") as f:
                self.move_stats = json.load(f)
            print(f"Loaded move statistics from {moves_path}")
        else:
            print(f"Note: No move statistics found at {moves_path}")

        return self.metrics is not None

    def generate_plots(self, figsize=(10, 6), dpi=100):
        """Generate and save plots of training metrics."""
        if self.metrics is None:
            print("No metrics loaded. Call load_metrics() first.")
            return False

        print(f"Generating plots in {self.output_dir}...")

        # Plot rewards
        plt.figure(figsize=figsize, dpi=dpi)
        if "episode_rewards" in self.metrics and "avg_rewards" in self.metrics:
            plt.plot(self.metrics["episode_rewards"], alpha=0.3, label="Episode Reward")
            plt.plot(self.metrics["avg_rewards"], label="Avg Reward")
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.title("Training Rewards")
            plt.legend()
            plt.savefig(os.path.join(self.output_dir, "rewards.png"))
            plt.close()
            print("Generated rewards plot")

        # Plot episode lengths
        plt.figure(figsize=figsize, dpi=dpi)
        if "episode_lengths" in self.metrics and "avg_episode_length" in self.metrics:
            plt.plot(self.metrics["episode_lengths"], alpha=0.3, label="Episode Length")
            plt.plot(self.metrics["avg_episode_length"], label="Avg Length")
            plt.xlabel("Episode")
            plt.ylabel("Steps")
            plt.title("Episode Lengths")
            plt.legend()
            plt.savefig(os.path.join(self.output_dir, "episode_lengths.png"))
            plt.close()
            print("Generated episode lengths plot")

        # Plot move accuracies
        plt.figure(figsize=figsize, dpi=dpi)
        if all(
            key in self.metrics
            for key in [
                "avg_perfect_accuracy",
                "avg_piece_accuracy",
                "avg_dest_accuracy",
            ]
        ):
            plt.plot(self.metrics["avg_perfect_accuracy"], label="Perfect Move")
            plt.plot(self.metrics["avg_piece_accuracy"], label="Piece Selection")
            plt.plot(self.metrics["avg_dest_accuracy"], label="Destination")
            plt.xlabel("Episode")
            plt.ylabel("Accuracy")
            plt.title("Move Accuracy")
            plt.legend()
            plt.savefig(os.path.join(self.output_dir, "move_accuracy.png"))
            plt.close()
            print("Generated move accuracy plot")

        # Plot game outcomes
        plt.figure(figsize=figsize, dpi=dpi)
        if all(key in self.metrics for key in ["win_rate", "draw_rate", "loss_rate"]):
            plt.plot(self.metrics["win_rate"], label="Win Rate")
            plt.plot(self.metrics["draw_rate"], label="Draw Rate")
            plt.plot(self.metrics["loss_rate"], label="Loss Rate")
            plt.xlabel("Episode")
            plt.ylabel("Rate")
            plt.title("Game Outcomes")
            plt.legend()
            plt.savefig(os.path.join(self.output_dir, "game_outcomes.png"))
            plt.close()
            print("Generated game outcomes plot")

        # If win/draw/loss rates are all 0, print a warning
        if (
            self.metrics.get("win_rate", [0])[-1] == 0
            and self.metrics.get("draw_rate", [0])[-1] == 0
            and self.metrics.get("loss_rate", [0])[-1] == 0
        ):
            print("\nWARNING: All game outcome rates (win/draw/loss) are 0.")
            print(
                "This could indicate an issue with how game outcomes are being tracked."
            )
            print("Check the code in train.py where results are recorded.\n")

        # Generate additional plots if move stats are available
        if self.move_stats:
            self._generate_move_stats_plots(figsize, dpi)

        print("All plots generated successfully")
        return True

    def _generate_move_stats_plots(self, figsize, dpi):
        """Generate additional plots based on move statistics."""
        # Extract episode numbers and perfect match percentages
        episodes = []
        perfect_matches = []
        piece_matches = []
        dest_matches = []

        # Group by episode
        episodes_data = {}
        for move in self.move_stats:
            ep = move["episode"]
            if ep not in episodes_data:
                episodes_data[ep] = {"perfect": 0, "piece": 0, "dest": 0, "total": 0}

            episodes_data[ep]["total"] += 1
            if move.get("perfect_match", False):
                episodes_data[ep]["perfect"] += 1
            if move.get("piece_match", False):
                episodes_data[ep]["piece"] += 1
            if move.get("dest_match", False):
                episodes_data[ep]["dest"] += 1

        # Convert to percentages per episode
        for ep in sorted(episodes_data.keys()):
            if episodes_data[ep]["total"] > 0:
                episodes.append(ep)
                perfect_matches.append(
                    episodes_data[ep]["perfect"] / episodes_data[ep]["total"]
                )
                piece_matches.append(
                    episodes_data[ep]["piece"] / episodes_data[ep]["total"]
                )
                dest_matches.append(
                    episodes_data[ep]["dest"] / episodes_data[ep]["total"]
                )

        # Plot move match percentages
        if episodes:
            plt.figure(figsize=figsize, dpi=dpi)
            plt.plot(episodes, perfect_matches, label="Perfect Match")
            plt.plot(episodes, piece_matches, label="Piece Match")
            plt.plot(episodes, dest_matches, label="Destination Match")
            plt.xlabel("Episode")
            plt.ylabel("Match Percentage")
            plt.title("Move Match Percentages")
            plt.legend()
            plt.savefig(os.path.join(self.output_dir, "move_matches.png"))
            plt.close()
            print("Generated move matches plot")

    def print_summary_statistics(self):
        """Print summary statistics from the training."""
        if self.metrics is None:
            print("No metrics loaded. Call load_metrics() first.")
            return

        print("\n===== TRAINING SUMMARY STATISTICS =====")

        # Episode information
        total_episodes = len(self.metrics.get("episode_rewards", []))
        print(f"Total Episodes: {total_episodes}")

        # Final metrics
        if "avg_rewards" in self.metrics and self.metrics["avg_rewards"]:
            print(f"Final Average Reward: {self.metrics['avg_rewards'][-1]:.2f}")

        if "avg_episode_length" in self.metrics and self.metrics["avg_episode_length"]:
            print(
                f"Final Average Episode Length: {self.metrics['avg_episode_length'][-1]:.1f} steps"
            )

        if (
            "avg_perfect_accuracy" in self.metrics
            and self.metrics["avg_perfect_accuracy"]
        ):
            print(
                f"Final Perfect Move Accuracy: {self.metrics['avg_perfect_accuracy'][-1]:.2%}"
            )

        if "avg_piece_accuracy" in self.metrics and self.metrics["avg_piece_accuracy"]:
            print(
                f"Final Piece Selection Accuracy: {self.metrics['avg_piece_accuracy'][-1]:.2%}"
            )

        if "avg_dest_accuracy" in self.metrics and self.metrics["avg_dest_accuracy"]:
            print(
                f"Final Destination Accuracy: {self.metrics['avg_dest_accuracy'][-1]:.2%}"
            )

        # Game outcomes
        if (
            all(key in self.metrics for key in ["win_rate", "draw_rate", "loss_rate"])
            and self.metrics["win_rate"]
        ):
            print(f"\nFinal Win Rate: {self.metrics['win_rate'][-1]:.2%}")
            print(f"Final Draw Rate: {self.metrics['draw_rate'][-1]:.2%}")
            print(f"Final Loss Rate: {self.metrics['loss_rate'][-1]:.2%}")


def main():
    """Main function to handle command-line arguments."""
    parser = argparse.ArgumentParser(description="Visualize Chess RL training metrics")
    parser.add_argument(
        "--metrics_dir",
        type=str,
        required=True,
        help="Directory containing training metrics JSON files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save output plots (default: metrics_dir/plots)",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=[10, 6],
        help="Figure size (width, height) in inches",
    )
    parser.add_argument("--dpi", type=int, default=100, help="DPI for saved figures")

    args = parser.parse_args()

    # Create visualizer
    visualizer = ChessMetricsVisualizer(args.metrics_dir, args.output_dir)

    # Load metrics
    if visualizer.load_metrics():
        # Generate plots
        visualizer.generate_plots(figsize=tuple(args.figsize), dpi=args.dpi)

        # Print summary
        visualizer.print_summary_statistics()
    else:
        print("Failed to load metrics. Check the provided directory path.")


if __name__ == "__main__":
    main()
