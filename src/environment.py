import random

import chess
import gym
import numpy as np
from gym import spaces

from utils import board_to_tensor, selections_to_move


class ChessEnv:
    def __init__(self):
        """Initialize a new chess environment"""
        self.reset()

    def reset(self):
        """Reset the environment to the starting position"""
        self.board = chess.Board()
        return self.board.copy()

    def step(self, action):
        """
        Execute an action (move) in the environment

        Args:
            action: chess.Move to execute

        Returns:
            tuple: (next_state, reward, done, info)
        """
        if action is None or action not in self.board.legal_moves:
            # Illegal move, penalize
            return self.board.copy(), -1.0, False, {"illegal_move": True}

        # Execute the move
        self.board.push(action)

        # Check if game is over
        done = self.board.is_game_over()

        # Simple reward structure (can be enhanced later)
        reward = 0.0

        # Check game result if done
        info = {}
        if done:
            result = self.board.result()
            if result == "1-0":  # White won
                reward = 1.0 if self.board.turn == chess.BLACK else -1.0
            elif result == "0-1":  # Black won
                reward = 1.0 if self.board.turn == chess.WHITE else -1.0
            info["result"] = result

        return self.board.copy(), reward, done, info

    def render(self):
        """Render the current board state (ASCII representation)"""
        return str(self.board)

    def get_legal_moves(self):
        """Get all legal moves in the current position"""
        return list(self.board.legal_moves)


class ChessPPOEnv(gym.Env):
    """
    Gym environment wrapper for chess that works with Stable Baselines3 PPO
    and includes action masking
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self, expert_model=None, opponent=None, include_masks=True, max_moves=200
    ):
        super(ChessPPOEnv, self).__init__()

        # Initialize the chess environment
        self.chess_env = ChessEnv()
        self.expert_model = expert_model

        self.include_masks = (
            include_masks  # Whether to include action masks in observation
        )
        self.max_moves = max_moves  # Maximum moves before declaring a draw

        # Define action space: 64 squares for piece selection + 64 possible moves
        self.action_space = spaces.MultiDiscrete([64, 64])

        # Define observation space based on whether we include masks
        if self.include_masks:
            self.observation_space = spaces.Dict(
                {
                    "board": spaces.Box(
                        low=0, high=1, shape=(12, 8, 8), dtype=np.float32
                    ),
                    "piece_mask": spaces.Box(
                        low=0, high=1, shape=(64,), dtype=np.float32
                    ),
                    "move_mask": spaces.Box(
                        low=0, high=1, shape=(64, 64), dtype=np.float32
                    ),
                }
            )
        else:
            # Simple observation space: just the board
            self.observation_space = spaces.Box(
                low=0, high=1, shape=(12, 8, 8), dtype=np.float32
            )

        self.current_board = None
        self.move_count = 0

    def reset(self):
        """Reset the environment to the starting position"""
        self.current_board = self.chess_env.reset()
        self.move_count = 0
        return self._get_observation()

    def _get_observation(self):
        """Convert chess board to the format expected by the policy network"""
        board_tensor = board_to_tensor(self.current_board).numpy()

        if not self.include_masks:
            return board_tensor

        # If including masks, create both piece and move masks
        valid_moves = list(self.current_board.legal_moves)

        # Create piece mask
        piece_mask = np.zeros(64, dtype=np.float32)

        # Group moves by source square
        moves_by_source = {}
        for move in valid_moves:
            source = move.from_square
            if source not in moves_by_source:
                moves_by_source[source] = []
            moves_by_source[source].append(move)

        # Fill in piece mask
        for source in moves_by_source:
            piece_mask[source] = 1.0

        # Create move mask for each piece
        move_mask = np.zeros((64, 64), dtype=np.float32)

        for source, moves in moves_by_source.items():
            # For each legal source square, mark valid target directions
            for i, move in enumerate(moves):
                if i < 64:  # Limit to 64 moves per piece
                    move_mask[source, i] = 1.0

        return {"board": board_tensor, "piece_mask": piece_mask, "move_mask": move_mask}

    def step(self, action):
        """
        Execute an action in the environment

        Args:
            action: Tuple of (piece_selection, move_selection)

        Returns:
            tuple: (observation, reward, done, info)
        """
        piece_selection, move_selection = action

        # Convert to a chess move
        valid_moves = list(self.current_board.legal_moves)
        chess_move = selections_to_move(piece_selection, move_selection, valid_moves)

        # If we couldn't find a valid move, choose randomly
        if chess_move is None and valid_moves:
            chess_move = random.choice(valid_moves)

        # Execute move in the chess environment
        next_board, env_reward, done, info = self.chess_env.step(chess_move)

        # Calculate reward (can use expert model if provided)
        reward = self._calculate_reward(next_board, chess_move)

        # Update state
        self.current_board = next_board
        self.move_count += 1

        # Check for move limit (avoid infinite games)
        if self.move_count >= self.max_moves:
            done = True
            info["result"] = "1/2-1/2"
            info["reason"] = "move_limit"

        if done:
            result = info.get("result", "1/2-1/2")
            if result == "1-0" and self.current_board.turn == chess.BLACK:
                # Current player (agent) won
                reward += 1.0
            elif result == "0-1" and self.current_board.turn == chess.WHITE:
                # Current player (agent) won
                reward += 1.0
            elif result == "1/2-1/2":
                # Draw
                reward += 0.5
            else:
                # Current player lost (shouldn't happen in single-player mode,
                # but included for completeness)
                reward -= 1.0

        # Get observation for next state
        observation = self._get_observation()

        return observation, reward, done, info

    def _calculate_reward(self, board, action):
        """
        Calculate reward for the current state and action

        Args:
            board: Current board state after the action
            action: The chess move that was executed

        Returns:
            float: Reward value
        """
        # Start with a small negative reward (encourage shorter games)
        reward = -0.01

        # If we have an expert model, use it for imitation learning
        if self.expert_model:
            expert_move = self.expert_model.get_best_move(board)

            # Expert-based reward
            if action == expert_move:
                reward += 1.0  # Perfect match
            elif expert_move:
                # Partial reward for similar moves
                from_square_match = action.from_square == expert_move.from_square
                to_square_match = action.to_square == expert_move.to_square

                if from_square_match:
                    reward += 0.5  # Selected the same piece
                elif to_square_match:
                    reward += 0.3  # Moved to the same square
                else:
                    # Check if the piece type is the same
                    piece_at_action = board.piece_at(action.from_square)
                    piece_at_expert = board.piece_at(expert_move.from_square)

                    if (
                        piece_at_action
                        and piece_at_expert
                        and piece_at_action.piece_type == piece_at_expert.piece_type
                    ):
                        reward += 0.1  # At least moved the same type of piece
                    else:
                        reward -= 0.1  # Different move

        return reward

    def render(self, mode="human"):
        """Render the current board state"""
        if mode == "human":
            return self.chess_env.render()
        return self.chess_env.render()

    def close(self):
        """Clean up resources"""
        pass


def create_environment(use_expert=True, include_masks=True):
    """Create a chess environment with optional expert model"""

    expert_model = None
    if use_expert:
        try:
            from expert import ChessExpert

            expert_model = ChessExpert()
        except (ImportError, Exception) as e:
            print(f"Could not initialize expert model: {e}")
            print("Training without expert guidance")

    # Create the environment
    env = ChessPPOEnv(expert_model=expert_model, include_masks=include_masks)

    return env
