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


class ChessPPOStockfishEnv(gym.Env):
    """
    Gym environment wrapper for chess that works with Stable Baselines3 PPO
    and includes action masking. Agent plays against Stockfish.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        expert_model=None,
        stockfish_opponent=None,
        stockfish_depth=1,  # Low depth for faster training
        include_masks=True,
        max_moves=200,
        agent_color=chess.WHITE,  # Agent plays as white by default
    ):
        super(ChessPPOStockfishEnv, self).__init__()

        # Initialize the chess environment
        self.chess_env = ChessEnv()
        self.expert_model = expert_model

        # Set up the Stockfish opponent
        self.stockfish_opponent = stockfish_opponent
        self.stockfish_depth = stockfish_depth

        # Set up which color the agent plays as
        self.agent_color = agent_color

        # Whether to include action masks in observation
        self.include_masks = include_masks

        # Maximum moves before declaring a draw
        self.max_moves = max_moves

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
        self.last_move = None  # Track the last move made

    def reset(self):
        """Reset the environment to the starting position"""
        self.current_board = self.chess_env.reset()
        self.move_count = 0
        self.last_move = None

        # If the agent is playing as black, let Stockfish make the first move
        if self.agent_color == chess.BLACK and self.stockfish_opponent:
            self._make_stockfish_move()

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

    def _make_stockfish_move(self):
        """Have Stockfish make a move on the current board"""
        if self.current_board.is_game_over():
            return False

        try:
            # Get move from Stockfish with the specified depth
            stockfish_move = self.stockfish_opponent.get_best_move(self.current_board)

            if stockfish_move is None:
                # If Stockfish returns None, choose a random move
                valid_moves = list(self.current_board.legal_moves)
                if valid_moves:
                    stockfish_move = random.choice(valid_moves)
                else:
                    return False

            # Execute Stockfish's move
            _, _, done, _ = self.chess_env.step(stockfish_move)
            self.current_board = self.chess_env.board.copy()
            self.move_count += 1
            self.last_move = stockfish_move
            return not done

        except Exception as e:
            print(f"Error in Stockfish move: {e}")
            # If there's an error, try a random move
            valid_moves = list(self.current_board.legal_moves)
            if valid_moves:
                random_move = random.choice(valid_moves)
                _, _, done, _ = self.chess_env.step(random_move)
                self.move_count += 1
                self.last_move = random_move
                return not done
            return False

    def step(self, action):
        """
        Execute an action in the environment and then let Stockfish respond

        Args:
            action: Tuple of (piece_selection, move_selection)

        Returns:
            tuple: (observation, reward, done, info)
        """
        # Ensure it's the agent's turn (should match the agent's color)
        if self.current_board.turn != self.agent_color:
            raise ValueError(
                f"Not the agent's turn. Board turn: {self.current_board.turn}, Agent color: {self.agent_color}"
            )

        piece_selection, move_selection = action

        # Convert to a chess move
        valid_moves = list(self.current_board.legal_moves)
        chess_move = selections_to_move(piece_selection, move_selection, valid_moves)

        # If we couldn't find a valid move, choose randomly
        if chess_move is None and valid_moves:
            chess_move = random.choice(valid_moves)

        # Execute the agent's move
        next_board, env_reward, done, info = self.chess_env.step(chess_move)
        self.current_board = next_board

        self.last_move = chess_move
        self.move_count += 1

        # Calculate immediate reward for the agent's move
        agent_move_reward = self._calculate_reward(next_board, chess_move)

        # Check for early game termination
        if self._check_termination(next_board, done, info):
            observation = self._get_observation()
            return observation, agent_move_reward, True, info

        # Now it's Stockfish's turn if the game isn't over
        if not done and self.stockfish_opponent:
            # Let Stockfish make a move
            stockfish_continued = self._make_stockfish_move()

            # Check if the game is over after Stockfish's move
            if not stockfish_continued or self.current_board.is_game_over():
                done = True
                if self.current_board.is_game_over():
                    result = self.current_board.result()
                    info["result"] = result

            # Re-calculate reward incorporating the result after Stockfish's move
            post_opponent_reward = self._calculate_post_opponent_reward(done, info)
            agent_move_reward += post_opponent_reward

        # Check for move limit (avoid infinite games)
        if self._check_termination(self.current_board, done, info):
            done = True

        # Get observation for next state (after both agent and Stockfish have moved)
        observation = self._get_observation()

        return observation, agent_move_reward, done, info

    def _check_termination(self, board, done, info):
        """Check if the game should terminate based on move limit or result"""
        # Check for move limit
        if self.move_count >= self.max_moves:
            info["result"] = "1/2-1/2"
            info["reason"] = "move_limit"
            return True

        return done

    def _calculate_reward(self, board, action):
        """
        Calculate immediate reward for the agent's move

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

        # Additional reward for capturing pieces
        to_square = action.to_square
        captured_piece = board.piece_at(to_square)
        if captured_piece:
            # Reward based on piece value
            piece_values = {
                chess.PAWN: 0.5,
                chess.KNIGHT: 0.7,
                chess.BISHOP: 0.7,
                chess.ROOK: 0.9,
                chess.QUEEN: 1.1,
            }
            reward += piece_values.get(captured_piece.piece_type, 0.0) * 0.1

        return reward

    def _calculate_post_opponent_reward(self, done, info):
        """
        Calculate the reward after the opponent's move

        Args:
            done: Whether the game is over
            info: Information about the game state

        Returns:
            float: Additional reward value
        """
        reward = 0.0

        # Check material balance
        material_value = self._evaluate_material_balance()
        reward += material_value * 0.05  # Scale the material advantage

        # If the game is over, add reward based on the outcome
        if done and "result" in info:
            result = info["result"]
            if result == "1-0" and self.agent_color == chess.WHITE:
                # Agent won
                reward += 3.0
            elif result == "0-1" and self.agent_color == chess.BLACK:
                # Agent won
                reward += 1.0
            elif result == "1/2-1/2":
                # Draw
                reward += 0.1
            else:
                # Agent lost
                reward -= 1.0

        return reward

    def _evaluate_material_balance(self):
        """
        Calculate the material balance of the position

        Returns:
            float: Material advantage from the agent's perspective
        """
        piece_values = {
            chess.PAWN: 0.5,
            chess.KNIGHT: 0.7,
            chess.BISHOP: 0.7,
            chess.ROOK: 0.9,
            chess.QUEEN: 1.1,
        }

        white_material = 0
        black_material = 0

        # Count material for each side
        for square in chess.SQUARES:
            piece = self.current_board.piece_at(square)
            if piece:
                value = piece_values.get(piece.piece_type, 0)
                if piece.color == chess.WHITE:
                    white_material += value
                else:
                    black_material += value

        # Return the material difference from the agent's perspective
        if self.agent_color == chess.WHITE:
            return white_material - black_material
        else:
            return black_material - white_material

    def render(self, mode="human"):
        """Render the current board state"""
        if mode == "human":
            return self.chess_env.render()
        return self.chess_env.render()

    def close(self):
        """Clean up resources"""
        if hasattr(self, "stockfish_opponent") and self.stockfish_opponent:
            try:
                self.stockfish_opponent.close()
            except:
                pass


def create_environment(
    use_expert=True,
    include_masks=True,
    stockfish_opponent=None,
    stockfish_depth=1,
    agent_color=chess.WHITE,
):
    """
    Create a chess environment with optional expert model and Stockfish opponent

    Args:
        use_expert: Whether to use an expert model for rewards
        include_masks: Whether to include action masks in observation
        stockfish_opponent: Optional pre-configured Stockfish object
        stockfish_depth: Depth for Stockfish search (if creating a new Stockfish)
        agent_color: Which color the agent plays as (chess.WHITE or chess.BLACK)

    Returns:
        ChessPPOStockfishEnv: The configured environment
    """
    expert_model = None
    if use_expert:
        try:
            from expert import ChessExpert

            expert_model = ChessExpert()
        except (ImportError, Exception) as e:
            print(f"Could not initialize expert model: {e}")
            print("Training without expert guidance")

    # Create or use the Stockfish opponent
    if stockfish_opponent is None:
        try:
            from expert import ChessExpert

            # Use a very low skill level and depth for faster training
            stockfish_opponent = ChessExpert(skill_level=1, depth=1)
            print(f"Created Stockfish opponent with depth {stockfish_depth}")
        except (ImportError, Exception) as e:
            print(f"Could not initialize Stockfish opponent: {e}")
            print("Training without Stockfish opponent")
            stockfish_opponent = None

    # Create the environment with the Stockfish opponent
    env = ChessPPOStockfishEnv(
        expert_model=expert_model,
        stockfish_opponent=stockfish_opponent,
        stockfish_depth=stockfish_depth,
        include_masks=include_masks,
        agent_color=agent_color,
    )

    return env
