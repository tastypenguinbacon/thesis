import os
import random

import numpy as np
from keras import Input
from keras import Model
from keras import Sequential
from keras.layers import Conv2D, Flatten, Dense, LeakyReLU, Dropout, Concatenate
from keras.optimizers import Adam
import tensorflow as tf
from keras import backend as K


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
        if batch_size * 8 > len(self.memory):
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
        neural_net.add(Conv2D(filters=64,
                              kernel_size=(3, 3),
                              input_shape=(height, width, 1)))
        neural_net.add(LeakyReLU(0.1))
        neural_net.add(Conv2D(128, (3, 3)))
        neural_net.add(LeakyReLU())
        neural_net.add(Conv2D(256, (3, 3)))
        neural_net.add(LeakyReLU(0.1))
        neural_net.add(Flatten())

        for i in range(4):
            neural_net.add(Dense(512, bias_initializer='ones'))
            neural_net.add(LeakyReLU(0.1))
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


class ActorCritic:
    def __init__(self, params):
        self.params = params
        self.sess = tf.Session()
        K.set_session(self.sess)
        self.action_size = params['cells_to_add'] * 2
        self.input_shape = params['height'], params['width'], 1

        self.memory = []
        self.actor_state_input, self.actor_model = self.create_actor_model()

        self.actor_critic_grad = tf.placeholder(tf.float32, [None, self.action_size])

        actor_model_weights = self.actor_model.trainable_weights

        self.actor_grads = tf.gradients(self.actor_model.output,
                                        actor_model_weights,
                                        -self.actor_critic_grad)

        grads = list(zip(self.actor_grads, actor_model_weights))

        self.optimize = tf.train.AdamOptimizer(learning_rate=0.0001, epsilon=0.01).apply_gradients(grads)

        self.critic_state_input, self.critic_action_input, self.critic_model = self.create_critic_model()

        self.critic_grads = tf.gradients(self.critic_model.output,
                                         self.critic_action_input)
        self.sess.run(tf.initialize_all_variables())

    def create_actor_model(self):
        state_input = Input(shape=self.input_shape)

        nnet = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(state_input)
        nnet = LeakyReLU(0.01)(nnet)
        nnet = Conv2D(filters=128, kernel_size=(3, 3), padding='same')(nnet)
        nnet = LeakyReLU(0.01)(nnet)
        nnet = Conv2D(filters=256, kernel_size=(3, 3), padding='same')(nnet)
        nnet = LeakyReLU(0.01)(nnet)
        nnet = Flatten()(nnet)

        for i in range(6):
            nnet = Dense(512)(nnet)
            nnet = LeakyReLU(0.01)(nnet)

        output = Dense(self.action_size, activation='sigmoid')(nnet)
        model = Model(input=state_input, output=output)
        model.compile(loss="mse", optimizer=Adam(lr=0.0001, epsilon=0.01, clipnorm=1.))

        return state_input, model

    def create_critic_model(self):
        state_input = Input(shape=self.input_shape)

        nnet = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(state_input)
        nnet = LeakyReLU(0.01)(nnet)
        nnet = Conv2D(filters=128, kernel_size=(3, 3), padding='same')(nnet)
        nnet = LeakyReLU(0.01)(nnet)
        nnet = Conv2D(filters=256, kernel_size=(3, 3), padding='same')(nnet)
        nnet = LeakyReLU(0.01)(nnet)
        nnet = Flatten()(nnet)

        action_input = Input(shape=(self.action_size,))
        action_h1 = Dense(512, bias_initializer='ones')(action_input)
        action_h1 = LeakyReLU(0.01)(action_h1)

        for i in range(4):
            action_h1 = Dense(512, bias_initializer='ones')(action_h1)
            action_h1 = LeakyReLU(0.01)(action_h1)

        merged_h1 = Concatenate()([nnet, action_h1])
        for i in range(6):
            merged_h1 = Dense(512, bias_initializer='ones')(merged_h1)
            merged_h1 = LeakyReLU(0.01)(merged_h1)

        output = Dense(1, activation='linear')(merged_h1)

        model = Model(input=[state_input, action_input], output=output)
        model.compile(loss="mse", optimizer=Adam(lr=0.0001, epsilon=0.01, clipnorm=1.))
        return state_input, action_input, model

    def remember(self, transition):
        state, action, reward, next_state = transition
        state = state.to_numpy_array()
        next_state = next_state.to_numpy_array()
        self.memory.append((state, self.__encode(action), reward, next_state))

    def learn(self):
        batch_size = self.params['batch_size']
        if len(self.memory) < batch_size * 4:
            return
        if len(self.memory) > batch_size * 8:
            self.memory = self.memory[::4]
        samples = random.sample(self.memory, batch_size)
        self._train_critic(samples)
        self._train_actor(samples)

    def _train_actor(self, samples):
        samples = list(zip(*samples))
        cur_state = np.array(samples[0])
        predicted_action = self.actor_model.predict(cur_state)
        grads = self.sess.run(self.critic_grads, feed_dict={
            self.critic_state_input: cur_state,
            self.critic_action_input: predicted_action
        })[0]

        self.sess.run(self.optimize, feed_dict={
            self.actor_state_input: cur_state,
            self.actor_critic_grad: grads
        })

    def _train_critic(self, samples):
        s, a, r, ns = list(zip(*samples))
        aa = self.actor_model.predict(np.array(ns))
        future_reward = self.critic_model.predict([np.array(ns), np.array(aa)]).flatten()
        r = np.array(r) + self.params['gamma'] * future_reward
        a = np.array(a)
        self.critic_model.fit([np.array(s), np.array(a)], r)

    def propose_action(self, cur_state, random=False):
        if np.random.random() < self.params['exploration_rate'] or random:
            action = np.random.rand(self.action_size)
        else:
            action = self.actor_model.predict(np.array([cur_state.to_numpy_array()]))[0]
            print('action', np.round(action, decimals=3))
        return self.__decode(action)

    def save(self, name):
        self.actor_model.save_weights('actor_' + name)
        self.critic_model.save_weights('critic_' + name)

    def load(self, name):
        if os.path.isfile('actor_' + name) and os.path.isfile('critic_' + name):
            self.actor_model.load_weights('actor_' + name)
            self.critic_model.load_weights('critic_' + name)

    def __encode(self, action):
        encoded = list(zip(*action))
        rows = list(np.array(encoded[0]).astype(np.float32) / (self.params['height'] - 1))
        cols = list(np.array(encoded[1]).astype(np.float32) / (self.params['width'] - 1))
        encoded = rows + cols
        return np.array(encoded)

    def __decode(self, action):
        rows = np.round(np.array(action[:len(action) // 2]) * (self.params['height'] - 1)).astype(np.int32)
        cols = np.round(np.array(action[len(action) // 2:]) * (self.params['width'] - 1)).astype(np.int32)
        ans = list(zip(rows, cols))
        return ans
