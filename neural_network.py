import os
import random

import matplotlib.pyplot as plt
import numpy as np
from keras import Sequential
from keras.layers import Conv2D, Flatten, Dense, Dropout
from keras.optimizers import Adam

from game_of_life import FocusArea, GameOfLife

width, height = 32, 32
focus_area = FocusArea(max_col=width, max_row=height)
number_of_epochs = 1000
game_iterations = 100
exploration_rate = 1
min_exploration_rate = 0.2

outputs_to_consider = 1
gamma = 0.5
replay_memory = []


class Reward:
    def __init__(self, min_cells, max_cells):
        self.min_cells = min_cells
        self.max_cells = max_cells

    def __call__(self, brd):
        count = len(brd)
        return 0.01 * (count - self.min_cells) * (count - self.max_cells)


neural_net = Sequential()
neural_net.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=(width, height, 1)))
print(neural_net.input_shape)
neural_net.add(Dropout(0.5))
neural_net.add(Conv2D(64, (3, 3), activation='relu'))
neural_net.add(Dropout(0.5))
neural_net.add(Conv2D(128, (3, 3), activation='relu'))
neural_net.add(Dropout(0.5))
neural_net.add(Conv2D(256, (3, 3), activation='relu'))
neural_net.add(Flatten())

neural_net.add(Dense(512, activation='relu'))
neural_net.add(Dropout(0.5))
neural_net.add(Dense(512, activation='relu'))
neural_net.add(Dropout(0.5))
neural_net.add(Dense(512, activation='relu'))
neural_net.add(Dropout(0.5))
neural_net.add(Dense(width * height + 1, activation='linear'))

neural_net.compile(optimizer=Adam(), loss='mse')


def random_board(size=None):
    how_many = np.random.randint(0, width * height) if size is None else size
    cols = np.random.randint(0, width, how_many)
    rows = np.random.randint(0, height, how_many)
    return zip(rows, cols)


def decode(outs):
    outs, = outs
    indices = np.argmax(outs)  # [-outputs_to_consider:]
    row, col = indices // width, indices % width
    return row, col


reward = Reward(60, 55)

if os.path.isfile('weights.be'):
    neural_net.load_weights('weights.be')

if __name__ == '__main__':
    for i in range(number_of_epochs):
        print('epoch', i)
        board = GameOfLife(focus_area, random_board())

        for j in range(game_iterations):
            if np.random.rand() < exploration_rate:
                action = tuple(random_board(outputs_to_consider))[0]
                exploration_rate *= 0.9999
                exploration_rate = max(exploration_rate, min_exploration_rate)
                is_random = True
            else:
                neural_net_in = np.array([board.to_numpy_array()])
                neural_net_out = neural_net.predict(neural_net_in)
                action = decode(neural_net_out)
                is_random = False

            print((i, j), len(board), action, is_random, sep='\t\t')
            next_board = board.add(action).next()
            r = reward(next_board)

            for_optimization = np.array([next_board.to_numpy_array()])
            replay_memory.append((board.to_numpy_array(), r, action[0] * width + action[1], for_optimization))
            board = next_board

            inputs, expected = [], []
            random_batch = random.sample(replay_memory, min(len(replay_memory), 256))
            previous_batch = replay_memory[max(-len(replay_memory), -16):]

            for s, r, a, ns in random_batch + previous_batch:
                inputs.append(s)
                prediction = neural_net.predict(ns)[0]
                e = r + gamma * prediction.max()
                prediction[int(a)] = e
                expected.append(prediction)

            neural_net.train_on_batch(np.array(inputs), np.array(expected))

        if i % 10 == 0:
            neural_net.save_weights('weights.be')
    plt.show()
