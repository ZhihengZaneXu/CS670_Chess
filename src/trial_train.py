import chess

from environment import create_environment
from expert import ChessExpert
from utils import create_chess_agent

# 1. Build env & agent (no learning)
env = create_environment(
    use_expert=False,
    include_masks=True,
    stockfish_opponent=ChessExpert(
        stockfish_path="/home/pd468/cs670/project/expert_models/stockfish/stockfish-ubuntu-x86-64-avx2",
        skill_level=1,
        depth=3,
    ),
    # <- supply a dummy opponent
    stockfish_depth=1,
    agent_color=chess.WHITE,
)
agent = create_chess_agent(env, gamma=0.99, n_steps=512, learning_rate=0.0003)

# 2. Reset
obs = env.reset()
print("Initial obs:", obs)

# 3. Step manually for N steps
for i in range(5):
    # agent.predict returns (action, _state)
    action, _ = agent.predict(obs, deterministic=False)
    print(f"Step {i:>2} ─ action selected:", action)

    # apply action
    obs, reward, done, info = env.step(action)
    print(f"reward={reward:.2f}, done={done}, info={info}")

    if done:
        break
