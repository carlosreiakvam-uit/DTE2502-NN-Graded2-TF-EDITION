# from game_environment_parallel import Snake
from agents.agent import DeepQLearningAgent

from game_environment import Snake
import numpy as np

env = Snake(board_size=10, frames=2)
s = env.reset()
done = 0
agent = DeepQLearningAgent(board_size=10, frames=2)
action = np.random.choice([-1, 0, 1], 1)[0]
# while not done:
#     board, reward, done, info, next_legal_moves = env.step(action)
#     action = np.random.choice(next_legal_moves[0], 1)[0]
#     env.print_game()
# print(info)

while not done:
    legal_moves = env.get_legal_moves()
    next_s, r, done, info, next_legal_moves = \
        agent.move(s, legal_moves, env.get_values())
    s = next_s.copy()
