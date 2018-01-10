import numpy as np

from agents import DQN
from game_of_life import FocusArea, GameOfLife

width, height = 8, 8
focus_area = FocusArea(max_col=width, max_row=height)
number_of_epochs = 1000
game_iterations = 200
exploration_rate = 0.2
gamma = 0.9


class Reward:
    def __init__(self, min_cells, max_cells):
        self.min_cells = min_cells
        self.max_cells = max_cells

    def __call__(self, brd, nxt, bad):
        mid = (self.min_cells + self.max_cells) / 2
        dif = (self.max_cells - self.min_cells) / 2
        cnt, bad_cnt, add = len(nxt), len(bad), 0
        if self.min_cells <= len(brd) <= self.max_cells and \
                (cnt < self.min_cells or cnt > self.max_cells):
            add = -10
        elif np.all(brd.to_numpy_array() == nxt.to_numpy_array()) and \
                (cnt < self.min_cells or cnt > self.max_cells):
            add = -10
        elif np.all(brd.to_numpy_array() == bad.to_numpy_array()) and \
                (cnt < self.min_cells or cnt > self.max_cells) and \
                        cnt == len(brd):
            add = -10
        elif cnt < self.min_cells:
            add = (cnt - bad_cnt) / 10
        elif cnt > self.max_cells:
            add = (bad_cnt - cnt) / 10
        else:
            add = 10
        return -np.abs(cnt - mid) + dif + add


class BinaryReward:
    def __init__(self, min_cells, max_cells):
        self.min_cells = min_cells
        self.max_cells = max_cells

    def __call__(self, brd, nxt, bad):
        if self.min_cells <= len(nxt) <= self.max_cells:
            return 5
        else:
            return -0.1


def random_board(size=None):
    how_many = np.random.randint(0, width * height) if size is None else size
    cols = np.random.randint(0, width, how_many)
    rows = np.random.randint(0, height, how_many)
    return zip(rows, cols)


def deep_q_learning():
    reward = BinaryReward(12, 16)
    name = 'cudo.be'
    nnet = DQN(params={'input_size': (height, width),
                       'exploration_probability': exploration_rate,
                       'batch_size': 2 ** 8,
                       'epochs': 1,
                       'learning_rate': gamma,
                       'max_mem': 2 ** 15})
    nnet.load(name)
    for i in range(number_of_epochs):
        board = GameOfLife(focus_area, random_board())
        for j in range(game_iterations):
            print(list(np.sort(nnet.neural_net.predict(np.array([board.to_numpy_array()]))[0])[-10:]))
            print(list(np.argsort(nnet.neural_net.predict(np.array([board.to_numpy_array()]))[0])[-10:]))
            print(board)
            if len(board) == 0:
                break
            bad_board = board.next()
            action = nnet.propose_action(board)
            next_board = board.add(*action).next()
            r = reward(board, next_board, bad_board)
            print((i, j), action, r, len(nnet.memory), len(board), '->', len(next_board))
            nnet.remember((board, action, r, next_board))
            board = next_board
            nnet.learn()
        nnet.save(name)


if __name__ == '__main__':
    deep_q_learning()
