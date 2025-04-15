"""Main module for Chess RL Agent training and evaluation."""

import os

import chess
import torch

from src.agent import ChessRLAgent
from src.config import get_args_parser
from src.environment import ChessEnv
from src.expert import ChessExpert
from src.utils import move_to_direction_idx


def train(args):
    """Train the RL agent"""
    # Create trained output directory if it doesn't exist
    os.makedirs(args.trained_output_dir, exist_ok=True)

    # Initialize expert
    expert = ChessExpert(args.stockfish_path)

    # Initialize RL agent
    agent = ChessRLAgent(
        expert,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
    )

    # Load existing pretrained model if specified
    if args.load_pretrained:
        print(f"Loading pretrained model from {args.load_pretrained}")
        agent.load(args.load_pretrained)

    # Train the agent
    total_timesteps = args.episodes * args.max_steps
    print(f"Training for {total_timesteps} total timesteps...")
    agent.train(total_timesteps=total_timesteps)

    # Save final model
    final_path = os.path.join(args.trained_output_dir, "chess_final_model")
    agent.save(final_path)
    print(f"Final model saved to {final_path}")

    # Close expert engine
    expert.close()


def evaluate(args):
    """Evaluate the trained RL agent"""
    # Initialize environment
    env = ChessEnv()

    # Initialize expert for comparison (optional)
    expert = None
    if args.compare_expert:
        expert = ChessExpert(args.stockfish_path)

    # Initialize agent
    agent = ChessRLAgent(expert=None)  # No expert needed for evaluation

    # Load model
    agent.load(args.load_model)
    agent.epsilon = 0.0  # No exploration during evaluation

    # Run evaluation games
    wins, draws, losses = 0, 0, 0

    for game in range(args.eval_games):
        state = env.reset()
        done = False
        step_count = 0

        while not done and step_count < args.max_steps:
            # Get agent's move
            action = agent.get_action(state)

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


def main():
    """Main entry point for the application."""
    parser = get_args_parser()
    args = parser.parse_args()

    if args.command == "train":
        train(args)
    elif args.command == "evaluate":
        evaluate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
