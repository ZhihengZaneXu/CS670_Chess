import chess
import gym
import numpy as np
from gym import spaces

from .utils import board_to_tensor


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

        # Simple reward structure (will be overridden by the agent's reward function)
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
    Custom gym environment wrapper for chess that works with Stable Baselines3 PPO
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, expert_model=None, opponent=None):
        super(ChessPPOEnv, self).__init__()

        # Initialize the chess environment
        self.chess_env = ChessEnv()
        self.expert_model = expert_model
        self.opponent = opponent  # Optional opponent for self-play

        # Define action and observation spaces required by gym
        # Action space: 64 squares for piece selection + 64 directions for move selection
        self.action_space = spaces.MultiDiscrete([64, 64])

        # Observation space: 12 channels (6 piece types × 2 colors) × 8×8 board
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(12, 8, 8), dtype=np.float32
        )

        self.current_board = None
        self.move_count = 0
        self.max_moves = 100  # Maximum moves before declaring a draw

    def reset(self):
        """
        Reset the environment to the starting position

        Returns:
            numpy.ndarray: The initial observation
        """
        self.current_board = self.chess_env.reset()
        self.move_count = 0
        return self._get_observation()

    def _get_observation(self):
        """Convert chess board to the format expected by the policy network"""
        return board_to_tensor(self.current_board).numpy()

    def step(self, action):
        """
        Execute an action in the environment

        Args:
            action: Tuple of (piece_selection, move_selection)

        Returns:
            tuple: (observation, reward, done, info)
        """
        from .utils import selections_to_move

        piece_selection, move_selection = action

        # Convert to a chess move
        valid_moves = list(self.current_board.legal_moves)
        chess_move = selections_to_move(piece_selection, move_selection, valid_moves)

        # If we couldn't find a valid move, choose randomly
        if chess_move is None and valid_moves:
            import random

            chess_move = random.choice(valid_moves)

        # Execute move in the chess environment
        next_board, _, done, info = self.chess_env.step(chess_move)

        # Get expert's move for reward calculation
        expert_move = None
        if self.expert_model:
            expert_move = self.expert_model.get_best_move(self.current_board)

        # Calculate reward based on expert imitation
        reward = self._calculate_reward(self.current_board, chess_move, expert_move)

        # Update state
        self.current_board = next_board
        self.move_count += 1

        # Check for move limit (avoid infinite games)
        if self.move_count >= self.max_moves:
            done = True
            info["result"] = "1/2-1/2"
            info["reason"] = "move_limit"

        # If playing against an opponent and game not over, let opponent make a move
        if not done and self.opponent:
            opponent_move = self.opponent.get_action(self.current_board)
            self.current_board, _, done, opp_info = self.chess_env.step(opponent_move)
            self.move_count += 1

            # Update info with opponent results if game ended
            if done:
                info.update(opp_info)

            # Adjust reward based on opponent's move outcome
            if done:
                result = info.get("result", "1/2-1/2")
                if result == "1-0" and self.current_board.turn == chess.BLACK:
                    reward += 1.0  # We won
                elif result == "0-1" and self.current_board.turn == chess.WHITE:
                    reward += 1.0  # We won
                elif result == "1/2-1/2":
                    reward += 0.5  # Draw is okay
                else:
                    reward -= 1.0  # We lost

        # Get observation for next state
        observation = self._get_observation()

        return observation, reward, done, info

    def _calculate_reward(self, board, action, expert_action):
        """
        Calculate reward based on similarity to expert action and game outcome

        Args:
            board: Current board state
            action: Agent's selected action
            expert_action: Expert's selected action

        Returns:
            float: Reward value
        """
        # Start with a small negative reward (encourage shorter games)
        reward = -0.01

        # If no expert or invalid moves, just return the small negative reward
        if action is None or expert_action is None:
            return reward

        # Reward for matching expert move
        if action == expert_action:
            reward += 1.0  # Perfect match
        else:
            # Partial reward for similar moves
            from_square_match = action.from_square == expert_action.from_square
            to_square_match = action.to_square == expert_action.to_square

            if from_square_match:
                reward += 0.5  # Selected the same piece
            elif to_square_match:
                reward += 0.3  # Moved to the same square
            else:
                # Check if the piece type is the same
                piece_at_action = board.piece_at(action.from_square)
                piece_at_expert = board.piece_at(expert_action.from_square)

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
        """
        Render the current board state

        Args:
            mode: The mode to render with

        Returns:
            str: The rendered board state
        """
        if mode == "human":
            return self.chess_env.render()
        else:
            return self.chess_env.render()

    def close(self):
        """Clean up resources"""
        pass
