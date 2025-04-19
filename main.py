"""Main module for Chess RL Agent training and evaluation."""

import argparse
import os

from src.environment import create_environment
from src.expert import ChessExpert
from src.network import create_chess_agent


def get_args_parser():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Chess RL Agent")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the agent")
    train_parser.add_argument(
        "--stockfish_path", type=str, default=None, help="Path to Stockfish executable"
    )
    train_parser.add_argument(
        "--episodes", type=int, default=1000, help="Number of training episodes"
    )
    train_parser.add_argument(
        "--max_steps", type=int, default=100, help="Maximum steps per episode"
    )
    train_parser.add_argument(
        "--gamma", type=float, default=0.99, help="Discount factor"
    )
    train_parser.add_argument(
        "--learning_rate", type=float, default=0.0003, help="Learning rate"
    )
    train_parser.add_argument(
        "--eval_interval", type=int, default=10, help="Episodes between evaluations"
    )
    train_parser.add_argument(
        "--save_interval",
        type=int,
        default=100,
        help="Episodes between saving checkpoints",
    )
    train_parser.add_argument(
        "--tensorboard_log",
        type=str,
        default="./logs/",
        help="Directory for TensorBoard logs",
    )
    train_parser.add_argument(
        "--output_dir",
        type=str,
        default="./trained_models/",
        help="Directory for saving models",
    )
    train_parser.add_argument(
        "--load_pretrained", type=str, default=None, help="Path to pretrained model"
    )

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate the agent")
    eval_parser.add_argument(
        "--load_model", type=str, required=True, help="Path to trained model"
    )
    eval_parser.add_argument(
        "--eval_games", type=int, default=10, help="Number of games to evaluate"
    )
    eval_parser.add_argument(
        "--max_steps", type=int, default=100, help="Maximum steps per game"
    )
    eval_parser.add_argument("--render", action="store_true", help="Render chess board")
    eval_parser.add_argument(
        "--stockfish_path",
        type=str,
        default=None,
        help="Path to Stockfish executable (for comparison)",
    )
    eval_parser.add_argument(
        "--compare_expert", action="store_true", help="Compare with expert"
    )

    return parser


def main():
    """Main entry point for the application."""
    parser = get_args_parser()
    args = parser.parse_args()

    if args.command == "train":
        # Import the training module here to avoid circular imports
        from train import train_chess_model

        train_chess_model(args)

    elif args.command == "evaluate":
        evaluate(args)

    else:
        parser.print_help()


def evaluate(args):
    """Evaluate the trained RL agent."""
    # Initialize environment
    env = create_environment(use_expert=False, include_masks=True)

    # Initialize expert for comparison (optional)
    expert = None
    if args.compare_expert:
        expert = ChessExpert(args.stockfish_path)

    # Load the trained agent
    agent = create_chess_agent(env, expert_model=None)
    agent = agent.load(args.load_model, env=env)

    # Run evaluation games
    wins, draws, losses = 0, 0, 0

    for game in range(args.eval_games):
        state = env.reset()
        done = False
        step_count = 0

        while not done and step_count < args.max_steps:
            # Get agent's action (deterministic for evaluation)
            action, _ = agent.predict(state, deterministic=True)

            # Take step in environment
            next_state, _, done, info = env.step(action)
            state = next_state
            step_count += 1

            # Print board state (optional)
            if args.render:
                print(env.render())
                print()

        # Game result
        if done:
            result = info.get("result", "1/2-1/2")
            if result == "1-0":
                wins += 1
            elif result == "0-1":
                losses += 1
            else:
                draws += 1

            print(f"Game {game+1}: Result = {result}, Moves = {step_count}")

    # Print overall statistics
    print(f"\nEvaluation Results ({args.eval_games} games):")
    print(f"Wins: {wins} ({wins/args.eval_games*100:.1f}%)")
    print(f"Draws: {draws} ({draws/args.eval_games*100:.1f}%)")
    print(f"Losses: {losses} ({losses/args.eval_games*100:.1f}%)")

    # Close expert if used
    if expert:
        expert.close()


if __name__ == "__main__":
    main()
