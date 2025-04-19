"""
Test module for Chess RL Agent that allows for evaluation against Stockfish
or use as a reference system during live play.
"""

import argparse
import time

import chess
import numpy as np
from stable_baselines3 import PPO

# Import necessary modules from your existing code
from environment import ChessEnv, create_environment
from expert import ChessExpert
from network import HierarchicalChessPolicy
from utils import board_to_tensor, create_chess_agent, selections_to_move


class ChessModelTester:
    """Class for testing trained Chess RL agent models."""

    def __init__(
        self,
        model_path,
        stockfish_path=None,
        stockfish_depth=5,  # Higher depth for testing
        stockfish_skill=10,  # Medium skill level for Stockfish
        agent_color=chess.WHITE,
    ):
        """Initialize the tester with a pretrained model."""
        self.model_path = model_path
        self.agent_color = agent_color

        # Initialize Stockfish
        self.stockfish = None
        if stockfish_path:
            try:
                self.stockfish = ChessExpert(
                    stockfish_path=stockfish_path,
                    skill_level=stockfish_skill,
                    depth=stockfish_depth,
                )
                print(
                    f"Initialized Stockfish with depth {stockfish_depth}, skill level {stockfish_skill}"
                )
            except Exception as e:
                print(f"Warning: Could not initialize Stockfish: {e}")
                print("Continuing without Stockfish opponent...")

        # Create environment without training components
        self.env = create_environment(
            use_expert=False,
            include_masks=True,
            stockfish_opponent=self.stockfish,
            stockfish_depth=stockfish_depth,
            agent_color=agent_color,
        )

        # Load the pretrained model
        self.agent = self._load_model(model_path)
        print(f"Successfully loaded model from {model_path}")

    def _load_model(self, model_path):
        """Load the pretrained model."""
        try:
            # Create a default agent first
            agent = create_chess_agent(self.env)

            # Now load the pretrained model
            agent = PPO.load(model_path, env=self.env)
            return agent
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def _get_agent_move(self, board, deterministic=True):
        """Get the agent's move recommendation for the current board state."""
        # Create observation from the current board
        if self.agent_color != board.turn:
            print(
                "Warning: It's not the agent's turn, but getting recommendation anyway"
            )

        # Convert board to the format expected by the policy
        if hasattr(self.env, "_get_observation"):
            # Use environment's method if available
            self.env.current_board = board
            obs = self.env._get_observation()
        else:
            # Manually create observation
            board_tensor = board_to_tensor(board).numpy()

            # Create a basic observation with masks
            valid_moves = list(board.legal_moves)

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

            # Create move mask
            move_mask = np.zeros((64, 64), dtype=np.float32)

            for source, moves in moves_by_source.items():
                for i, move in enumerate(moves):
                    if i < 64:  # Limit to 64 moves per piece
                        move_mask[source, i] = 1.0

            obs = {
                "board": board_tensor,
                "piece_mask": piece_mask,
                "move_mask": move_mask,
            }

        # Get action from the agent
        action, _ = self.agent.predict(obs, deterministic=deterministic)

        # Convert action to chess move
        piece_selection, move_selection = action
        valid_moves = list(board.legal_moves)
        chess_move = selections_to_move(piece_selection, move_selection, valid_moves)

        # If we couldn't find a valid move, choose randomly
        if chess_move is None and valid_moves:
            import random

            chess_move = random.choice(valid_moves)

        return chess_move

    def play_automated_game(self, num_games=1, render=True):
        """Play full automated games against Stockfish and report results."""
        if self.stockfish is None:
            print("Error: Stockfish is not available. Cannot play automated games.")
            return

        results = {"wins": 0, "losses": 0, "draws": 0}

        for game_num in range(1, num_games + 1):
            print(f"\nStarting Game {game_num}/{num_games}")

            # Reset the environment
            board = chess.Board()
            turn_count = 0
            game_over = False

            while (
                not game_over and turn_count < 200
            ):  # Limit to 200 moves to prevent infinite games
                turn_count += 1

                # Determine whose turn it is
                if board.turn == self.agent_color:
                    # Agent's turn
                    move = self._get_agent_move(board)
                    if move is None:
                        print("Agent couldn't find a valid move. Resigning.")
                        results["losses"] += 1
                        game_over = True
                        break

                    board.push(move)
                    if render:
                        print(f"\nTurn {turn_count}, Agent's move: {move.uci()}")
                        print(board)
                else:
                    # Stockfish's turn
                    start_time = time.time()
                    stockfish_move = self.stockfish.get_best_move(board)
                    end_time = time.time()

                    if stockfish_move is None:
                        print("Stockfish couldn't find a valid move. Agent wins.")
                        results["wins"] += 1
                        game_over = True
                        break

                    board.push(stockfish_move)
                    if render:
                        print(
                            f"\nTurn {turn_count}, Stockfish's move: {stockfish_move.uci()} (took {end_time - start_time:.2f}s)"
                        )
                        print(board)

                # Check if the game is over
                if board.is_game_over():
                    game_over = True
                    result = board.result()

                    if result == "1-0" and self.agent_color == chess.WHITE:
                        print("Agent (White) wins!")
                        results["wins"] += 1
                    elif result == "0-1" and self.agent_color == chess.BLACK:
                        print("Agent (Black) wins!")
                        results["wins"] += 1
                    elif result == "1-0" and self.agent_color == chess.BLACK:
                        print("Stockfish (White) wins!")
                        results["losses"] += 1
                    elif result == "0-1" and self.agent_color == chess.WHITE:
                        print("Stockfish (Black) wins!")
                        results["losses"] += 1
                    else:
                        print("Game ended in a draw!")
                        results["draws"] += 1

            # If we reached move limit
            if not game_over:
                print("Game reached move limit (200). Declaring a draw.")
                results["draws"] += 1

        # Print overall results
        print("\n===== OVERALL RESULTS =====")
        print(f"Games played: {num_games}")
        print(f"Agent wins: {results['wins']} ({results['wins']/num_games:.1%})")
        print(f"Draws: {results['draws']} ({results['draws']/num_games:.1%})")
        print(f"Agent losses: {results['losses']} ({results['losses']/num_games:.1%})")

        return results

    def play_interactive_game(self):
        """
        Play an interactive game where the human plays against Stockfish
        with the agent providing move recommendations.
        """
        board = chess.Board()
        turn_count = 0
        game_over = False

        # Determine if human is playing as white or black
        human_color = chess.BLACK if self.agent_color == chess.WHITE else chess.WHITE
        human_color_name = "Black" if human_color == chess.BLACK else "White"
        agent_color_name = "White" if self.agent_color == chess.WHITE else "Black"

        print("\n===== INTERACTIVE CHESS GAME =====")
        print(f"You are playing as {human_color_name}")
        print(f"AI agent will provide recommendations as {agent_color_name}")
        print("Enter moves in UCI format (e.g., e2e4)")
        print("Type 'help' for available commands")
        print("Initial board:")
        print(board)

        stockfish_active = self.stockfish is not None

        while not game_over:
            turn_count += 1

            if board.turn == human_color:
                # Human's turn
                print(f"\nTurn {turn_count} - Your move ({human_color_name})")

                # Get AI recommendation
                agent_recommendation = self._get_agent_move(board)
                if agent_recommendation:
                    print(f"AI recommends: {agent_recommendation.uci()}")

                # If Stockfish is available, get its recommendation too
                if stockfish_active:
                    stockfish_recommendation = self.stockfish.get_best_move(board)
                    if stockfish_recommendation:
                        print(f"Stockfish recommends: {stockfish_recommendation.uci()}")

                # Display legal moves
                legal_moves = list(board.legal_moves)
                print(
                    f"Legal moves: {', '.join(move.uci() for move in legal_moves[:10])}"
                    + (
                        f" (and {len(legal_moves) - 10} more...)"
                        if len(legal_moves) > 10
                        else ""
                    )
                )

                # Get human input
                while True:
                    user_input = (
                        input("\nYour move (or 'help', 'board', 'resign', 'auto'): ")
                        .strip()
                        .lower()
                    )

                    if user_input == "help":
                        print("\nCommands:")
                        print("  <move> - Enter a move in UCI format (e.g., e2e4)")
                        print("  help - Show this help message")
                        print("  board - Show the current board")
                        print("  resign - Resign the game")
                        print("  auto - Let the AI make a move for you")
                        print("  legal - Show all legal moves")
                        continue

                    if user_input == "board":
                        print(board)
                        continue

                    if user_input == "resign":
                        print("You resigned. Game over.")
                        game_over = True
                        break

                    if user_input == "auto":
                        # Use the AI recommendation
                        if agent_recommendation:
                            move = agent_recommendation
                            print(f"Using AI recommendation: {move.uci()}")
                            board.push(move)
                            print(board)
                            break
                        else:
                            print(
                                "AI couldn't find a recommendation. Please enter a move manually."
                            )
                            continue

                    if user_input == "legal":
                        print("Legal moves:")
                        for i, move in enumerate(board.legal_moves):
                            print(f"{move.uci()}", end=" ")
                            if (i + 1) % 8 == 0:
                                print()  # New line every 8 moves
                        print("\n")
                        continue

                    # Try to parse as a move
                    try:
                        move = chess.Move.from_uci(user_input)
                        if move in board.legal_moves:
                            board.push(move)
                            print(board)
                            break
                        else:
                            print("Illegal move. Try again.")
                    except ValueError:
                        print("Invalid format. Enter move in UCI format (e.g., e2e4)")

            else:
                # Stockfish's turn
                print(f"\nTurn {turn_count} - Stockfish's move ({agent_color_name})")

                if stockfish_active:
                    start_time = time.time()
                    stockfish_move = self.stockfish.get_best_move(board)
                    end_time = time.time()

                    if stockfish_move is None:
                        print("Stockfish couldn't find a valid move. You win!")
                        game_over = True
                        break

                    # Let's also get the agent's recommendation for comparison
                    agent_recommendation = self._get_agent_move(board)
                    if agent_recommendation:
                        print(
                            f"Note: AI would have played: {agent_recommendation.uci()}"
                        )

                    board.push(stockfish_move)
                    print(
                        f"Stockfish plays: {stockfish_move.uci()} (took {end_time - start_time:.2f}s)"
                    )
                    print(board)
                else:
                    print("No Stockfish available. Using agent as opponent.")
                    agent_move = self._get_agent_move(board)

                    if agent_move is None:
                        print("Agent couldn't find a valid move. You win!")
                        game_over = True
                        break

                    board.push(agent_move)
                    print(f"Agent plays: {agent_move.uci()}")
                    print(board)

            # Check if the game is over
            if board.is_game_over():
                game_over = True
                result = board.result()

                if result == "1-0" and human_color == chess.WHITE:
                    print("You win as White!")
                elif result == "0-1" and human_color == chess.BLACK:
                    print("You win as Black!")
                elif result == "1-0" and human_color == chess.BLACK:
                    print("Stockfish wins as White!")
                elif result == "0-1" and human_color == chess.WHITE:
                    print("Stockfish wins as Black!")
                else:
                    print("Game ended in a draw!")

                print(f"Final board state after {turn_count} turns:")
                print(board)


def main():
    """Main function to parse arguments and run the tester."""
    parser = argparse.ArgumentParser(description="Test a trained chess RL agent")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the pretrained model"
    )
    parser.add_argument(
        "--stockfish_path", type=str, default=None, help="Path to Stockfish executable"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["auto", "interactive"],
        default="interactive",
        help="Testing mode: 'auto' for automated games, 'interactive' for live play",
    )
    parser.add_argument(
        "--num_games",
        type=int,
        default=5,
        help="Number of games to play in automated mode",
    )
    parser.add_argument(
        "--stockfish_depth",
        type=int,
        default=5,
        help="Depth for Stockfish search (1-20)",
    )
    parser.add_argument(
        "--stockfish_skill",
        type=int,
        default=10,
        help="Skill level for Stockfish (0-20)",
    )
    parser.add_argument(
        "--agent_color",
        type=str,
        default="white",
        choices=["white", "black"],
        help="Color for the agent to play as",
    )

    args = parser.parse_args()

    # Convert agent color string to chess.Color
    agent_color = chess.WHITE if args.agent_color.lower() == "white" else chess.BLACK

    # Create the tester
    tester = ChessModelTester(
        model_path=args.model_path,
        stockfish_path=args.stockfish_path,
        stockfish_depth=args.stockfish_depth,
        stockfish_skill=args.stockfish_skill,
        agent_color=agent_color,
    )

    # Run the selected mode
    if args.mode == "auto":
        tester.play_automated_game(num_games=args.num_games)
    else:
        tester.play_interactive_game()


if __name__ == "__main__":
    main()
