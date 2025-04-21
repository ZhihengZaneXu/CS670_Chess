import chess
import numpy as np
from stockfish import Stockfish

from utils import board_to_tensor, selections_to_move


class ChessExpert:

    def __init__(
        self,
        stockfish_path="/home/pd468/cs670/project/expert_models/stockfish/stockfish-ubuntu-x86-64-avx2",
        skill_level=0,
        depth=5,
    ):
        """
        Initialize the chess expert (mentor) model using Stockfish package

        Args:
            stockfish_path: Path to Stockfish executable
            skill_level: Stockfish skill level (0-20)
            depth: Search depth for Stockfish
        """
        # Initialize Stockfish engine
        self.stockfish = Stockfish(
            path=stockfish_path, depth=depth, parameters={"Skill Level": skill_level}
        )

    def get_best_move(self, board, time_limit=None):
        """
        Get the best move from the expert

        Args:
            board: chess.Board representation of the current state
            time_limit: Time in seconds to think (optional)

        Returns:
            chess.Move: The best move according to the expert
        """
        # Set the position in Stockfish
        self.stockfish.set_fen_position(board.fen())

        # Set time limit if provided (in milliseconds)
        if time_limit is not None:
            self.stockfish.set_depth(0)  # Reset depth
            self.stockfish.update_engine_parameters(
                {"Move Time": int(time_limit * 1000)}
            )

        # Get the best move
        best_move_uci = self.stockfish.get_best_move()

        # If no legal moves or Stockfish can't find a move
        if best_move_uci is None:
            # Return a random legal move if available
            legal_moves = list(board.legal_moves)
            if legal_moves:
                import random

                return random.choice(legal_moves)
            return None

        # Convert UCI string to chess.Move
        return chess.Move.from_uci(best_move_uci)

    def close(self):
        """Close the engine process (not needed with stockfish package)"""
        # The stockfish package handles engine process cleanup automatically
        pass

        from environment import board_to_tensor, selections_to_move


class ModelOpponent:
    def __init__(self, model, include_masks=True):
        """
        Wraps a trained SB3 model so it looks like a `ChessExpert`

        Args:
            model:   a Stable‑Baselines3 agent (with .predict())
            include_masks: whether to build piece/move masks
        """
        self.model = model
        self.include_masks = include_masks

    def get_best_move(self, board):
        """
        Given a python-chess Board, return the move chosen
        by the wrapped policy in UCI form.
        """
        # 1) get raw board tensor
        board_tensor = board_to_tensor(board).numpy()

        # 2) optionally build masks exactly as in the env
        if not self.include_masks:
            obs = board_tensor
        else:
            valid_moves = list(board.legal_moves)
            # group legal moves by source square
            moves_by_source = {}
            for mv in valid_moves:
                moves_by_source.setdefault(mv.from_square, []).append(mv)

            # piece mask: which from‑squares are legal
            piece_mask = np.zeros(64, dtype=np.float32)
            for src in moves_by_source:
                piece_mask[src] = 1.0

            # move mask: for each src, which move‑slots 0–63 are valid
            move_mask = np.zeros((64, 64), dtype=np.float32)
            for src, mvs in moves_by_source.items():
                for i, mv in enumerate(mvs):
                    if i < 64:
                        move_mask[src, i] = 1.0

            obs = {
                "board": board_tensor,
                "piece_mask": piece_mask,
                "move_mask": move_mask,
            }

        # 3) ask the policy for its action (deterministic)
        action, _ = self.model.predict(obs, deterministic=True)

        # 4) map (piece_sel, move_sel) → chess.Move
        legal = list(board.legal_moves)
        chosen_move = selections_to_move(int(action[0]), int(action[1]), legal)

        return chosen_move
