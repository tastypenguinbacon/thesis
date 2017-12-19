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
exploration_rate = 1
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
            return - (count_next - self.min_cells) * (count_next - self.max_cells) + (count - count_next) ** 2
        return - (count_next - self.min_cells) * (count_next - self.max_cells)


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
        if np.all(state != next_state) or reward > 0:
            self.replay_memory.append((state, reward, action, next_state))

    def replay(self):
        self.i += 1
        if self.i % 128 != 0:
            return
        inputs, expected = [], []
        random_batch = random.sample(self.replay_memory, min(len(self.replay_memory), 256))
        prev_sample = self.replay_memory[-min(128, len(self.replay_memory)):]
        print(len(self.replay_memory))
        for s, r, a, ns in random_batch + prev_sample:
            inputs.append(s)
            prediction = self.neural_net.predict(np.array([ns]))[0]
            best_prediction = prediction.max()
            e = r + gamma * best_prediction
            prediction[int(a)] = e
            expected.append(prediction)

        if len(inputs) != 0:
            self.neural_net.fit(np.array(inputs), np.array(expected))


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


nets = [
    (SingleNet(height + 1), SingleNet(width + 1)),
    (SingleNet(height + 1), SingleNet(width + 1)),
    (SingleNet(height + 1), SingleNet(width + 1)),
    (SingleNet(height + 1), SingleNet(width + 1)),
    (SingleNet(height + 1), SingleNet(width + 1))
]

reward = Reward(55, 60)

for i, nnet_pair in enumerate(nets):
    nnet_pair[0].load('row' + str(i) + '.be')
    nnet_pair[1].load('col' + str(i) + '.be')

if __name__ == '__main__':
    for i in range(number_of_epochs):
        exp_rate = exploration_rate
        if np.random.rand() < 0.9:
            board = GameOfLife(focus_area)
        else:
            board = GameOfLife(focus_area, random_board())

        for j in range(game_iterations):
            if np.random.rand() < exp_rate:
                action = tuple(random_board(len(nets)))
                exp_rate *= 0.99
                exp_rate = max(exp_rate, min_exploration_rate)
            else:
                action = []
                temp_board = board
                for row_nnet, col_nnet in nets:
                    a = (row_nnet.predict(temp_board), col_nnet.predict(temp_board))
                    action.append(a)
                    temp_board = temp_board.add(a)

            print((i, j), len(board), action, sep='\t\t')
            # print("-" * 16)
            # print(board)
            next_board = board

            short_time_mem = []

            for (row_nnet, col_nnet), a in zip(nets, action):
                numpy_board = next_board.to_numpy_array()
                next_board = next_board.add(a)
                short_time_mem.append((numpy_board, a, next_board.to_numpy_array()))
            # print("-" * 16)
            # print(next_board)
            # print("-" * 16)
            next_board = next_board.next()
            r = reward(board, next_board)
            print(len(board), r, sep='\t')
            board = next_board

            for (row_nnet, col_nnet), (prev_board, act, next_board) in zip(nets, short_time_mem):
                row_nnet.remember(prev_board, r, act[0], next_board)
                col_nnet.remember(prev_board, r, act[1], next_board)

            for row_nnet, col_nnet in nets:
                row_nnet.replay()
                col_nnet.replay()

        for i, nnet_pair in enumerate(nets):
            nnet_pair[0].save('row' + str(i) + '.be')
            nnet_pair[1].save('col' + str(i) + '.be')
