import os
import random
from collections import deque

import numpy as np
from keras import Input
from keras import Model
from keras import Sequential
from keras.layers import Conv2D, Flatten, Dense, Add, LeakyReLU, Dropout


class DQN:
    def __init__(self, params):
        self.memory = []
        self.input_size = params['input_size']
        self.neural_net = self.__create_net()
        self.params = params
        self.for_opt_arange = np.arange(params['batch_size'])

    def propose_action(self, board):
        if random.random() < self.params['exploration_probability']:
            height, width = self.input_size
            return [(np.random.randint(height), np.random.randint(width))]
        nnet_in = np.array([board.to_numpy_array()])
        nnet_out = self.neural_net.predict(nnet_in)[0]
        return [self.__decode(np.argmax(nnet_out))]

    def remember(self, transition):
        state, action, reward, next_state = transition
        state = state.to_numpy_array()
        next_state = next_state.to_numpy_array()
        self.memory.append((state, self.__encode(action), reward, next_state))

    def learn(self):
        if self.params['max_mem'] < len(self.memory):
            self.memory = self.memory[::2]
        batch_size = self.params['batch_size']
        if batch_size * 4 > len(self.memory):
            return
        batch = random.sample(self.memory, batch_size)

        batch = list(zip(*batch))
        states, actions, rewards, next_states = batch
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        next_predictions = self.neural_net.predict(next_states)
        next_max = np.max(next_predictions, axis=1)
        prediction = self.neural_net.predict(states)
        to_set = rewards + self.params['learning_rate'] * next_max
        prediction[self.for_opt_arange, actions] = to_set
        self.neural_net.fit(states, prediction, verbose=1, epochs=self.params['epochs'])

    def __create_net(self):
        height, width = self.input_size
        neural_net = Sequential()
        neural_net.add(Conv2D(filters=32,
                              kernel_size=(3, 3),
                              input_shape=(height, width, 1)))
        neural_net.add(LeakyReLU(0.1))
        neural_net.add(Conv2D(64, (3, 3)))
        neural_net.add(LeakyReLU())
        neural_net.add(Conv2D(128, (3, 3)))
        neural_net.add(LeakyReLU(0.1))
        neural_net.add(Flatten())

        for i in range(3):
            neural_net.add(Dense(256, bias_initializer='ones'))
            neural_net.add(LeakyReLU(0.1))
            neural_net.add(Dropout(0.75))
        neural_net.add(Dense(height * width + 1, activation='linear'))
        neural_net.compile(optimizer='adam', loss='mse')
        return neural_net

    def __decode(self, number):
        height, width = self.input_size
        return number // width, number % width

    def __encode(self, action):
        _, width = self.input_size
        (row, col), = action
        return row * width + col

    def save(self, name):
        self.neural_net.save_weights(name)

    def load(self, name):
        if os.path.isfile(name):
            self.neural_net.load_weights(name)


class AC:
    def __init__(self, params):
        self.params = params
        self.memory = deque(maxlen=2000)

    def __actor(self):
        state_input = Input(shape=self.params.observation_space.shape)
        h1 = Dense(24, activation='relu')(state_input)
        h2 = Dense(48, activation='relu')(h1)
        h3 = Dense(24, activation='relu')(h2)
        output = Dense(self.params.action_space.shape[0],
                       activation='relu')(h3)
        model = Model(input=state_input, output=output)
        model.compile(optimizer='rmsprop', loss='mse')
        return state_input, model

    def __critic(self):
        state_input = Input(shape=self.env.observation_space.shape)
        state_h1 = Dense(24, activation='relu')(state_input)
        state_h2 = Dense(48)(state_h1)

        action_input = Input(shape=self.env.action_space.shape)
        action_h1 = Dense(48)(action_input)

        merged = Add()([state_h2, action_h1])
        merged_h1 = Dense(24, activation='relu')(merged)
        output = Dense(1, activation='relu')(merged_h1)
        model = Model(input=[state_input, action_input],
                      output=output)

        model.compile(optimizer='rmsprop', loss='mse')
        return state_input, action_input, model
