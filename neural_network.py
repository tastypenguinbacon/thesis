import os
import random

import numpy as np
from keras import Sequential
from keras.layers import Conv2D, Flatten, Dense, Dropout
from keras.optimizers import Adam

from game_of_life import FocusArea, GameOfLife

width, height = 16, 16
focus_area = FocusArea(max_col=width, max_row=height)
number_of_epochs = 1000
game_iterations = 512
exploration_rate = 0.8
min_exploration_rate = 0.01
learning_rate = 0.5
gamma = 0.9


class Reward:
    def __init__(self, min_cells, max_cells):
        self.min_cells = min_cells
        self.max_cells = max_cells

    def __call__(self, brd, nxt):
        count = len(brd)
        count_next = len(nxt)
        if count < count_next < self.min_cells or count > count_next > self.max_cells:
            return np.abs(count_next - count)
        if self.min_cells < count_next < self.max_cells:
            return 100
        if count_next - count == 0:
            return -1
        return -np.abs(count_next - count) * 1


class SingleNet:
    def __init__(self, size):
        neural_net = Sequential()
        neural_net.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=(width, height, 1),
                              bias_initializer='ones'))
        neural_net.add(Dropout(0.5))
        neural_net.add(Conv2D(64, (3, 3), activation='relu', bias_initializer='ones'))
        neural_net.add(Dropout(0.5))
        neural_net.add(Conv2D(128, (3, 3), activation='relu', bias_initializer='ones'))
        neural_net.add(Dropout(0.5))
        neural_net.add(Conv2D(256, (3, 3), activation='relu', bias_initializer='ones'))
        neural_net.add(Dropout(0.5))
        neural_net.add(Conv2D(512, (3, 3), activation='relu', bias_initializer='ones'))
        neural_net.add(Dropout(0.5))
        neural_net.add(Conv2D(1024, (3, 3), activation='relu', bias_initializer='ones'))
        neural_net.add(Flatten())

        for i in range(6):
            neural_net.add(Dense(256, activation='relu', bias_initializer='ones'))
            neural_net.add(Dropout(0.5))
        neural_net.add(Dense(size, activation='linear'))

        neural_net.compile(optimizer=Adam(), loss='mse')

        self.neural_net = neural_net
        self.replay_memory = []

        self.i = 0

    def predict(self, board):
        neural_net_in = np.array([board.to_numpy_array()])
        neural_net_out = self.neural_net.predict(neural_net_in)[0]
        return np.argmax(neural_net_out)

    def load(self, name):
        if os.path.isfile(name):
            self.neural_net.load_weights(name)

    def save(self, name):
        self.neural_net.save_weights(name)

    def remember(self, state, reward, action, next_state):
        self.replay_memory.append((state, reward, action, next_state))

    def replay(self):
        inputs, expected = [], []
        random_batch = random.sample(self.replay_memory, min(len(self.replay_memory), 128))
        prev_sample = self.replay_memory[-min(64, len(self.replay_memory)):]

        for s, r, a, ns in random_batch + prev_sample:
            inputs.append(s)
            prediction = self.neural_net.predict(np.array([ns]))[0]
            best_prediction = prediction.max()
            e = r + gamma * best_prediction
            prediction[int(a)] = e
            expected.append(prediction)

        if len(inputs) != 0:
            print(len(self.replay_memory))
            self.neural_net.fit(np.array(inputs), np.array(expected))


def random_board(size=None):
    how_many = np.random.randint(0, width * height) if size is None else size
    cols = np.random.randint(0, width, how_many)
    rows = np.random.randint(0, height, how_many)
    return zip(rows, cols)


def decode(outs):
    # outs, = outs
    row, col = outs // width, outs % width
    return row, col


def encode(action):
    return action[0] * width + action[1]


nnet = SingleNet(width * height + 1)

reward = Reward(55, 60)

with open('cudo.json', 'a') as out_file:
    nnet.load('q_nnet.be')
    if __name__ == '__main__':
        for i in range(number_of_epochs):
            exp_rate = exploration_rate
            board = GameOfLife(focus_area, random_board())
            print('[', end='', file=out_file, flush=True)
            for j in range(game_iterations):
                if np.random.rand() < exp_rate:
                    action = tuple(random_board(1))[0]
                    exp_rate *= 0.99
                    exp_rate = max(exp_rate, min_exploration_rate)
                    is_rand = True
                else:
                    cudo = nnet.predict(board)
                    action = decode(cudo)
                    print(cudo, action)
                    is_rand = False
                next_board = board.add(action).next()
                # print(len(board), end=',', file=out_file)
                # print("-" * 16)
                # print(board)
                # print("-" * 16)
                # print(next_board)
                # print("-" * 16)
                r = reward(board, next_board)
                print((i, j), len(board), action, r, is_rand, sep='\t\t')
                nnet.remember(board.to_numpy_array(), r, encode(action), next_board.to_numpy_array())
                board = next_board
                # print(len(board), r, sep='\t')
                nnet.replay()

            print('],', file=out_file, flush=True)
            nnet.save('q_nnet.be')
