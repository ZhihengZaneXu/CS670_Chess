import unittest

import chess
import torch

from src.environment import ChessEnv
from src.expert import ChessExpert
from src.network import ChessNetwork, ChessRLAgent
from src.utils import board_to_tensor, move_to_direction_idx, selections_to_move


class TestChessRL(unittest.TestCase):
    def test_board_to_tensor(self):
        """Test board_to_tensor conversion"""
        board = chess.Board()
        tensor = board_to_tensor(board)

        # Check shape
        self.assertEqual(tensor.shape, (12, 8, 8))

        # Check if pawns are correctly placed
        # White pawns on the second rank
        for col in range(8):
            self.assertEqual(tensor[0, 1, col].item(), 1.0)

        # Black pawns on the seventh rank
        for col in range(8):
            self.assertEqual(tensor[6, 6, col].item(), 1.0)

    def test_move_to_direction_idx(self):
        """Test move_to_direction_idx function"""
        board = chess.Board()

        # Test pawn move
        e4 = chess.Move.from_uci("e2e4")
        idx = move_to_direction_idx(e4)
        self.assertEqual(idx, 0)  # North direction, 2 squares

        # Test knight move
        knight_move = chess.Move.from_uci("b1c3")
        idx = move_to_direction_idx(knight_move)
        self.assertEqual(idx, 60)  # Knight move encoding

    def test_selections_to_move(self):
        """Test selections_to_move function"""
        board = chess.Board()
        valid_moves = list(board.legal_moves)

        # Test selecting e2 pawn and moving it forward 2 squares
        piece_selection = chess.E2
        direction_selection = 0  # North direction, 2 squares

        move = selections_to_move(piece_selection, direction_selection, valid_moves)
        self.assertEqual(move, chess.Move.from_uci("e2e4"))

    def test_chess_network(self):
        """Test ChessNetwork forward pass"""
        network = ChessNetwork()
        board = chess.Board()
        tensor = board_to_tensor(board).unsqueeze(0)  # Add batch dimension

        piece_logits, move_logits = network(tensor)

        # Check output shapes
        self.assertEqual(piece_logits.shape, (1, 64))
        self.assertEqual(move_logits.shape, (1, 64))

    def test_chess_env(self):
        """Test ChessEnv"""
        env = ChessEnv()
        board = env.reset()

        # Check initial state
        self.assertTrue(isinstance(board, chess.Board))
        self.assertEqual(board.fen(), chess.STARTING_FEN)

        # Test step function
        e4 = chess.Move.from_uci("e2e4")
        next_state, reward, done, info = env.step(e4)

        # Check that the move was executed
        self.assertEqual(next_state.piece_at(chess.E4).piece_type, chess.PAWN)
        self.assertEqual(next_state.piece_at(chess.E4).color, chess.WHITE)
        self.assertFalse(done)

    @unittest.skip("Requires Stockfish to be installed")
    def test_expert(self):
        """Test ChessExpert (requires Stockfish)"""
        try:
            expert = ChessExpert()
            board = chess.Board()

            # Get expert move
            move = expert.get_best_move(board)

            # Check that it returns a valid move
            self.assertTrue(move in board.legal_moves)

            expert.close()
        except ValueError:
            self.skipTest("Stockfish not found")

    def test_agent_init(self):
        """Test ChessRLAgent initialization"""

        # Create a mock expert
        class MockExpert:
            def get_best_move(self, board):
                return list(board.legal_moves)[0]

        agent = ChessRLAgent(MockExpert())

        # Check that models are initialized
        self.assertTrue(isinstance(agent.model, ChessNetwork))
        self.assertTrue(isinstance(agent.target_model, ChessNetwork))


if __name__ == "__main__":
    unittest.main()
