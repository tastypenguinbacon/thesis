# from tkinter import Tk
#
# from game_of_life import GameOfLife, FocusArea
# import numpy as np
# import matplotlib.pyplot as plt
#
# from neural_network import neural_net, decode, random_board
#
# from gui import GameOfLifeBoard, Dim
#
# width, height = 16, 16
# focus_area = FocusArea(max_col=width, max_row=height)
#
# brd = GameOfLife(FocusArea(width, height), random_board())
#
# for i in range(height):
#     brd = brd.add((i, i))
#     brd = brd.add((height - i - 1, i))
#     brd = brd.add((i, height // 2))
#     brd = brd.add((i, height // 2 - 1))
#     brd = brd.add((width // 2, i))
#     brd = brd.add((width // 2 - 1, i))
#
# GameOfLifeBoard.pixels_per_box = 10
# GameOfLifeBoard.inter_frame_time = 250
#
#
#
# class Mutator:
#     def __call__(self, brd):
#         action = neural_net.predict(np.array(brd.to_numpy_array()))[0]
#         return brd.add(decode(action))
#
#
#
# root = Tk()
# root.title('Conway\'s game of Life')
#
# canv = GameOfLifeBoard(root, Dim(height, width), brd)  # , Mutator(width, height))
#
# canv.tick()
#
# root.mainloop()
#
# plt.plot(canv.cells)
# plt.grid(True)
# plt.axis([0, len(canv.cells) - 1, 0, max(canv.cells)])
# plt.show()

from correct_agents import DeepQNetwork
from correct_game_of_life import GameOfLife

import numpy as np


class DecoderEncoder:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols

    def decode(self, number):
        return number // self.cols, number % self.cols

    def encode(self, cell):
        row, col = cell
        return row * self.cols + col


class BinaryReward:
    def __init__(self, min_cells, max_cells):
        self.min_cells = min_cells
        self.max_cells = max_cells

    def __call__(self, board):
        if self.min_cells <= len(board) <= self.max_cells:
            return 1
        else:
            return -0.1


def random_board(height, width, size=None):
    how_many = np.random.randint(width * height // 4, width * height) if size is None else size
    cols = np.random.randint(0, width, how_many)
    rows = np.random.randint(0, height, how_many)
    return zip(rows, cols)


if __name__ == '__main__':
    rows, cols = 8, 6
    number_of_games = 10000
    game_iterations = 200

    file_name = 'cudo_%d_%d.h5' % (rows, cols)

    dqn = DeepQNetwork(
        input_size=(rows, cols, 1),
        min_batch_size=32,
        batch_size=128,
        exploration_probability=0.9,
        learning_rate=0.9,
        max_mem=1024,
    )

    dqn.load(file_name)

    decenc = DecoderEncoder(rows, cols)
    reward = BinaryReward(2, 2)

    for i in range(number_of_games):
        gol = GameOfLife(max_row=rows, max_col=cols, game_of_life=random_board(rows, cols))
        for j in range(game_iterations):
            gol.print()
            s = gol.numpy_array()
            a = dqn.action(s)
            cells_to_add = list(map(lambda x: decenc.decode(x), a))
            gol = gol.next(cells_to_add)
            sn = gol.numpy_array()
            if gol.count() == 0:
                break
            r = reward(gol.game_of_life)
            dqn.remember((s, a, r, sn))
            print('game: {0} step: {1} reward: {2}, action: {3}, cells: {4}'.format(i, j, r, decenc.decode(a[0]), gol.count()))
            dqn.learn()
        dqn.save(file_name)

