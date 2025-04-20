import argparse
import datetime
import os
from collections import Counter

import chess
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)

from environment import create_environment
from expert import ModelOpponent
from utils import FrequentLoggingCallback, create_chess_agent, selections_to_move


class CustomMetricsCallback(BaseCallback):
    """Log W/D/L rates and average piece captures per game."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.wins = self.draws = self.losses = self.total_games = 0

        # Track total captures across all games for calculating averages
        self.total_captures_by_type = {
            chess.PAWN: 0,
            chess.KNIGHT: 0,
            chess.BISHOP: 0,
            chess.ROOK: 0,
            chess.QUEEN: 0,
        }

        # Track captures for the current game
        self.current_game_captures = {
            chess.PAWN: 0,
            chess.KNIGHT: 0,
            chess.BISHOP: 0,
            chess.ROOK: 0,
            chess.QUEEN: 0,
        }

        # Keep track of the previous board state to detect captures
        self.prev_board_state = None
        self.total_moves = 0
        self.perfect_moves = 0
        self.correct_piece_selections = 0
        self.correct_move_selections = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        dones = self.locals.get("dones")
        actions = self.locals.get("actions")
        if infos is None or dones is None or actions is None:
            return True

        # Get current board state
        current_board = self.training_env.envs[0].current_board

        # If we have a previous board state, check for captures
        if self.prev_board_state is not None:
            prev_pieces = self.prev_board_state.piece_map()
            current_pieces = current_board.piece_map()

            # Find pieces that were on the board before but are now gone
            for square, piece in prev_pieces.items():
                if (
                    square not in current_pieces
                    or current_pieces[square].piece_type != piece.piece_type
                ):
                    # A piece was captured or replaced (e.g., pawn promotion)
                    # We're only interested in captures here, and kings aren't captured in chess
                    if (
                        square not in current_pieces
                        or current_pieces[square].color != piece.color
                    ) and piece.piece_type != chess.KING:
                        # Only count captures made by the agent (based on piece color)
                        agent_color = self.training_env.envs[0].agent_color
                        piece_color = piece.color

                        # If the agent is capturing opponent pieces
                        if piece_color != agent_color:
                            self.current_game_captures[piece.piece_type] += 1

        # Update the previous board state for the next step
        self.prev_board_state = current_board.copy()

        for info, done, action in zip(infos, dones, actions):
            expert_move = info.get("expert_move")
            env = self.training_env.envs[0]

            # Check move matching only if expert_move is provided
            if expert_move:
                agent_move = selections_to_move(
                    int(action[0]), int(action[1]), list(env.current_board.legal_moves)
                )

                if agent_move:
                    self.total_moves += 1

                    # Perfect move (exact match)
                    if agent_move == expert_move:
                        self.perfect_moves += 1

                    # Correct piece selection (from square match)
                    if agent_move.from_square == expert_move.from_square:
                        self.correct_piece_selections += 1

                    # Correct move selection (to square match)
                    if agent_move.to_square == expert_move.to_square:
                        self.correct_move_selections += 1

            if not done or "result" not in info:
                continue

            # Update W/D/L statistics
            self.total_games += 1
            result = info["result"]
            color = self.training_env.envs[0].agent_color
            if result == "1-0":
                if color == chess.WHITE:
                    self.wins += 1
                else:
                    self.losses += 1
            elif result == "0-1":
                if color == chess.BLACK:
                    self.wins += 1
                else:
                    self.losses += 1
            else:
                self.draws += 1

            # Update total captures with the current game's captures
            for piece_type, count in self.current_game_captures.items():
                self.total_captures_by_type[piece_type] += count

            # Calculate and log averages
            if self.total_games > 0:
                avg_pawn = self.total_captures_by_type[chess.PAWN] / self.total_games
                avg_knight = (
                    self.total_captures_by_type[chess.KNIGHT] / self.total_games
                )
                avg_bishop = (
                    self.total_captures_by_type[chess.BISHOP] / self.total_games
                )
                avg_rook = self.total_captures_by_type[chess.ROOK] / self.total_games
                avg_queen = self.total_captures_by_type[chess.QUEEN] / self.total_games

                self.logger.record("histogram/pawns", avg_pawn)
                self.logger.record("histogram/knights", avg_knight)
                self.logger.record("histogram/bishops", avg_bishop)
                self.logger.record("histogram/rooks", avg_rook)
                self.logger.record("histogram/queens", avg_queen)

            # Log standard W/D/L rates
            self.logger.record("custom/win_rate", self.wins / self.total_games)
            self.logger.record("custom/draw_rate", self.draws / self.total_games)
            self.logger.record("custom/loss_rate", self.losses / self.total_games)

            # Log perfect move rate
            if self.total_moves > 0:
                self.logger.record(
                    "quality/perfect_move_rate", self.perfect_moves / self.total_moves
                )
                self.logger.record(
                    "quality/correct_piece_selection_rate",
                    self.correct_piece_selections / self.total_moves,
                )
                self.logger.record(
                    "quality/correct_move_selection_rate",
                    self.correct_move_selections / self.total_moves,
                )

            # Reset move counts after logging
            self.total_moves = self.perfect_moves = self.correct_piece_selections = (
                self.correct_move_selections
            ) = 0

            # Reset for the next game
            self.current_game_captures = {
                chess.PAWN: 0,
                chess.KNIGHT: 0,
                chess.BISHOP: 0,
                chess.ROOK: 0,
                chess.QUEEN: 0,
            }
            self.prev_board_state = None

        return True


def set_seeds(seed=42):
    """Set seeds for reproducibility"""
    import random

    import numpy as np
    import torch

    # Set Python's random seed
    random.seed(seed)

    # Set NumPy's random seed
    np.random.seed(seed)

    # Set PyTorch's random seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Set additional PyTorch settings for determinism
    # Note: This may impact performance
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variables (some libraries check these)
    import os

    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    parser = argparse.ArgumentParser(description="Train a hierarchical chess RL agent")

    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--episodes", type=int, default=1000, help="Number of episodes to train"
    )
    parser.add_argument(
        "--max_steps", type=int, default=500, help="Max steps per episode"
    )
    parser.add_argument(
        "--stockfish_path",
        type=str,
        default="../expert_models/stockfish/stockfish-ubuntu-x86-64-avx2",
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
    agent_color = chess.WHITE if args.agent_color == chess.WHITE else chess.BLACK
    env = create_environment(
        use_expert=bool(args.stockfish_path),
        include_masks=True,
        stockfish_opponent=None,
        stockfish_depth=args.stockfish_depth,
        agent_color=agent_color,
    )

    set_seeds(args.seed)
    # agent = create_chess_agent(
    #     env,
    #     gamma=args.gamma,
    #     n_steps=args.n_steps,
    #     learning_rate=args.learning_rate,
    #     tensorboard_log=args.tensorboard_log,
    # )

    agent = create_chess_agent(
        env,
        gamma=0.99,
        n_steps=512,
        learning_rate=3e-4,
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

    env.stockfish_opponent = ModelOpponent(agent, include_masks=True)
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
