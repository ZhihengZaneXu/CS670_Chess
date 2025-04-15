import chess
from stockfish import Stockfish


class ChessExpert:

    def __init__(
        self,
        stockfish_path=None,
        skill_level=20,
        depth=15,
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
