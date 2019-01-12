import random
from os.path import isfile

import numpy as np
from keras import Sequential
from keras.layers import Conv2D, Flatten, Dense


class DeepQNetwork:
    def __init__(self, **params):
        self.params = params
        self.memory = []
        self.neural_network = self.__create_net()
        self.for_opt_arange = np.arange(params['batch_size'])

    def __create_net(self):
        net = Sequential()
        net.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=self.params['input_size']))
        for i in range(3):
            net.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        net.add(Flatten())
        net.add(Dense(self.params['input_size'][0] * self.params['input_size'][1] + 1))
        net.compile("adam", "mse")
        return net

    def action(self, board):
        if random.random() < self.params['exploration_probability']:
            self.params['exploration_probability'] *= 0.99
            height, width, _ = self.params['input_size']
            return [np.random.randint(height * width + 1)]

        net_in = np.array([board])
        net_out = self.neural_network.predict(net_in)[0]
        return [np.argmax(net_out)]

    def remember(self, transition):
        s, a, r, sn = transition
        self.memory.append((s, a[0], r, sn))

    def learn(self):
        if self.params['batch_size'] > len(self.memory):
            return
        if self.params['max_mem'] < len(self.memory):
            self.memory = random.sample(self.memory, self.params['max_mem'] // 2)
        batch = random.sample(self.memory, self.params['batch_size'])
        batch = {s.tostring(): (s, a, r, ns) for (s, a, r, ns) in batch}
        batch = batch.values()

        if len(batch) < self.params['min_batch_size']:
            return

        s, a, r, ns = list(map(np.array, list(zip(*batch))))

        Q_ns = self.neural_network.predict(ns)
        Q_ns_max = np.max(Q_ns, axis=1)
        Q = self.neural_network.predict(s)
        Q[self.for_opt_arange[:len(batch)], a] = r + self.params['learning_rate'] * Q_ns_max
        self.neural_network.fit(s, Q, verbose=1)

    def save(self, name):
        self.neural_network.save_weights(name)

    def load(self, name):
        if isfile(name):
            self.neural_network.load_weights(name)


class ActorCritic:
    def __init__(self):
        pass

    def action(self):
        pass

    def remember(self):
        pass

    def learn(self):
        pass

    def save(self, name):
        pass

    def load(self, name):
        pass
