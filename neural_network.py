import os
import random
from operator import itemgetter

import numpy as np
from keras import Sequential
from keras.layers import Conv2D, Flatten, Dense, Dropout, LeakyReLU

from game_of_life import FocusArea, GameOfLife

width, height = 8, 8
focus_area = FocusArea(max_col=width, max_row=height)
number_of_epochs = 10000
game_iterations = 1000
exploration_rate = 0.2
cells_to_add = 5
gamma = 0.5


class Reward:
    def __init__(self, min_cells, max_cells):
        self.min_cells = min_cells
        self.max_cells = max_cells

    def __call__(self, brd, nxt, bad):
        # count = len(brd)
        count_next = len(nxt)
        count = len(bad)
        if count < count_next < self.min_cells or count > count_next > self.max_cells:
            return np.abs(count_next - count)
        if self.min_cells < count_next < self.max_cells:
            return - (count_next - self.min_cells) * (count_next - self.max_cells)
        return -np.abs(count_next - count)


class SingleNet:
    def __init__(self, size):
        neural_net = Sequential()
        neural_net.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(width, height, 1)))
        neural_net.add(LeakyReLU())
        neural_net.add(Dropout(0.5))
        neural_net.add(Conv2D(64, (3, 3)))
        neural_net.add(LeakyReLU())
        neural_net.add(Dropout(0.5))
        neural_net.add(Flatten())

        for i in range(1):
            neural_net.add(Dense(128, bias_initializer='ones'))
            neural_net.add(LeakyReLU())
            neural_net.add(Dropout(0.5))
        neural_net.add(Dense(size, activation='linear'))

        neural_net.compile(optimizer='rmsprop', loss='mse')

        self.neural_net = neural_net
        self.replay_memory = []

        self.i = 0

    def predict(self, board):
        neural_net_in = np.array([board.to_numpy_array()])
        neural_net_out = self.neural_net.predict(neural_net_in)[0]
        if np.isnan(neural_net_out).any():
            print('NANANANANANNANANA BATMAN')
        print('min, max: ', np.min(neural_net_out), np.max(neural_net_out))
        return np.argmax(neural_net_out)

    def predict_batch(self, board, size):
        neural_net_in = np.array([board.to_numpy_array()])
        neural_net_out = self.neural_net.predict(neural_net_in)[0]
        a = np.argsort(neural_net_out)[-size:]
        b = np.sort(neural_net_out)[-size:]
        print('\n'.join(map(str, zip(a, b))))
        return np.argsort(neural_net_out)[-size:]

    def load(self, name):
        if os.path.isfile(name):
            self.neural_net.load_weights(name)

    def save(self, name):
        self.neural_net.save_weights(name)

    def remember(self, state, reward, action, next_state):
        if np.all(state == next_state) and reward > 0:
            self.replay_memory.append((state, 1000, action, next_state))
        self.replay_memory.append((state, reward, action, next_state))

    def replay(self):
        inputs, expected = [], []
        random_batch = random.sample(self.replay_memory, min(len(self.replay_memory), 1024))

        for s, r, a, ns in random_batch:
            inputs.append(s)
            prediction = self.neural_net.predict(np.array([ns]))[0]
            best_prediction = prediction.max()
            e = r + gamma * best_prediction
            prediction[int(a)] = e
            expected.append(prediction)

        if len(inputs) != 0:
            self.neural_net.train_on_batch(np.array(inputs), np.array(expected))  # , verbose=0)
            print(len(self.replay_memory))


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

reward = Reward(16, 20)

with open('cudo.json', 'a') as out_file:
    nnet.load('q_nnet.be')
    if __name__ == '__main__':
        for i in range(number_of_epochs):
            exp_rate = exploration_rate
            board = GameOfLife(focus_area, random_board())
            print('[', end='', file=out_file)
            for j in range(game_iterations):
                print(len(board), end=',', file=out_file)
                if len(board) == 0:
                    continue
                bad_board, next_board, action, r = board.next(), None, None, None
                if np.random.rand() < exp_rate:
                    best_actions = []
                    for action in nnet.predict_batch(board, width * height):
                        action = decode(action)
                        nb = board.add(action).next()
                        r = reward(board, nb, bad_board)
                        best_actions.append((r, action, nb))
                    best_actions.sort(key=itemgetter(0))
                    best_actions = best_actions[-width:]
                    for r, action, nb in best_actions:
                        nnet.remember(board.to_numpy_array(), r, encode(action), nb.to_numpy_array())
                    is_random = True
                else:
                    nnet.predict_batch(board, 5)
                    a = nnet.predict(board)
                    print(a)
                    action = decode(a)
                    is_random = False

                next_board = board.add(action).next()
                r = reward(board, next_board, bad_board)
                nnet.remember(board.to_numpy_array(), r, encode(action), next_board.to_numpy_array())
                print((i, j), len(board), action, r, is_random, sep='\t\t')
                print(board)
                board = next_board
                nnet.replay()

            print('],', file=out_file, flush=True)
            nnet.save('q_nnet.be')
