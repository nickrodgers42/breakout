import gym
import tensorflow as tf
import random
import numpy as np
import os
import json

from replay_memory import ReplayMemory
from frame_processor import FrameProcessor
from model import Model

ENV_NAME = 'BreakoutDeterministic-v4'

class Breakout():
    def __init__(self, session, memory_size, memory_start_size, frame_buff_size, batch_size, 
        learning_rate, update_frequency, no_op_max, max_episode_length, 
        network_update_frequency, max_epsilon, min_epsilon, gamma, render=True):

        self._session = session

        self._env = gym.make(ENV_NAME)
        self._fp = FrameProcessor()
        self._memory = ReplayMemory(
            size=memory_size,
            frame_height=self._fp._height,
            frame_width=self._fp._width,
            frame_buff_size=frame_buff_size,
            batch_size=batch_size
        )
        with tf.variable_scope('main_dqn'):
            self._main_dqn = Model(
                input_shape=[None, self._fp._height, self._fp._width, frame_buff_size],
                num_outputs=self._env.action_space.n,
                learning_rate=learning_rate
            )
        with tf.variable_scope('target_dqn'):
            self._target_dqn = Model(
                input_shape=[None, self._fp._height, self._fp._width, frame_buff_size],
                num_outputs=self._env.action_space.n,
                learning_rate=learning_rate
            )
        self._main_dqn_vars = tf.trainable_variables(scope='main_dqn')
        self._target_dqn_vars = tf.trainable_variables(scope='target_dqn')

        self._var_init = tf.global_variables_initializer()
        self._saver = tf.train.Saver()

        self._frame_buff_size = frame_buff_size
        self._memory_max_size = memory_size
        self._memory_start_size = memory_start_size
        self._update_frequency =  update_frequency
        self._no_op_max = no_op_max
        self._max_episode_length = max_episode_length
        self._network_update_frequency = network_update_frequency
        self._render = render

        self._max_epsilon = max_epsilon
        self._min_epsilon = min_epsilon
        self._epsilon = self._max_epsilon
        self._slope = -(self._max_epsilon - self._min_epsilon) / self._memory_max_size
        self._intercept = self._max_epsilon
        self._gamma = gamma

        self._state = []
        self._last_lives = 5
        self._episode_length = 0
        self._steps = 0
        self._total_reward = 0
        self._total_loss = 0
        
        self._saver = tf.train.Saver()

    def _fill_memory(self):
        while self._memory.size < self._memory_start_size:
            self._env.reset()
            no_ops = random.randint(0, self._no_op_max)
            while True:
                if self._render:
                    self._env.render()
                action = self._env.action_space.sample()
                frame, reward, gameover, _ = self._env.step(action)
                frame = self._fp.preprocess(frame)
                if no_ops > 0:
                    no_ops -= 1
                else:
                    self._memory.add_memory((frame, action, reward, gameover))
                if gameover or self._memory.size >= self._memory_start_size:
                    break
            print(f'Filling Memory... Size: {self._memory.size}', end='\r')
        print('\nMemory Filled')

    def _reset_game_state(self):
        self._last_lives = 5
        self._episode_length = 0
        action = 0
        frame = self._env.reset()
        frame = self._fp.preprocess(frame)
        frame_buff = [frame]
        while len(frame_buff) < self._frame_buff_size:
            frame = self._env.step(action)[0]
            frame = self._fp.preprocess(frame)
            frame_buff.append(frame)
        self._state = frame_buff
        self._total_loss = 0
        self._total_reward = 0

    def _update_game_state(self, frame):
        self._state.append(frame)
        self._state.pop(0)
    
    def _update_state(self, state, frame):
        state = np.split(state, self._frame_buff_size, 2)
        state.append(frame)
        state.pop(0)
        return np.dstack(state)

    def _play_choose_action(self):
        state = np.dstack(self._state,)
        if random.random() < 0.1:
            return self._env.action_space.sample()
        action = self._main_dqn.get_best_action(
            self._session,
            state
        )[0]
        print(f'Action: {action}')
        return action

    def _choose_action(self):
        self._epsilon = self._slope * self._steps + self._intercept
        self._epsilon = max(self._epsilon, self._min_epsilon)
        if self._epsilon == self._min_epsilon and self._min_epsilon != 0:
            self._slope = - self._min_epsilon / self._memory_max_size
            self._intercept = self._min_epsilon - (self._slope * self._memory_max_size) 
            self._min_epsilon = 0
        if random.random() < self._epsilon:
            return self._env.action_space.sample()
        state = np.dstack(self._state, )
        action = self._main_dqn.get_best_action(
            self._session,
            state
        )[0]
        # print(f'Action: {action}')
        return action

    def play(self):
        self._reset_game_state()
        # if self._memory.size < self._memory_start_size:
        #     self._fill_memory()
        while True:
            if self._render:
                self._env.render()
            action = self._play_choose_action()
            frame, reward, gameover, _ = self._env.step(action)
            frame = self._fp.preprocess(frame)
            self._update_game_state(frame)
            reward = np.sign(reward)
            
            self._steps += 1
            self._episode_length += 1
            self._total_reward += reward
            self._print_training_status()
            
            if gameover or self._episode_length > self._max_episode_length:
                break
        print()
        return self._total_reward

    def train(self):
        self._reset_game_state()
        no_ops = random.randint(0, self._no_op_max)
        action = 1

        if self._memory.size < self._memory_start_size:
            self._fill_memory()

        while True:
            if self._render:
                self._env.render()
            if no_ops > 0:
                no_ops -= 1
            else:
                action = self._choose_action()
            
            frame, reward, gameover, _ = self._env.step(action)
            frame = self._fp.preprocess(frame)
            self._update_game_state(frame)
            reward = np.sign(reward)

            if no_ops <= 0:
                self._memory.add_memory((frame, action, reward, gameover))
                self._steps += 1
                self._episode_length += 1
                self._total_reward += reward
                if self._steps % self._update_frequency == 0:
                    self._total_loss += self._learn()
                if self._steps % self._network_update_frequency == 0:
                    self._update_networks()
                self._print_training_status()

            if gameover or self._episode_length > self._max_episode_length:
                break
        print()
        return self._total_reward, self._total_loss

    def _print_training_status(self):
        print(f'Step: {self._steps}, '
            f'Total reward: {self._total_reward}, '
            f'Epsilon: {self._epsilon:.4f}, '
            f'Loss: {self._total_loss:.4f}, '
            f'Stored frames: {self._memory.size}', 
            end='\r'
        )

    def _learn(self):
        memory_batch = self._memory.sample()
        states = np.array([memory[0] for memory in memory_batch])
        actions = np.array([memory[1] for memory in memory_batch])
        rewards = np.array([memory[2] for memory in memory_batch])
        next_states = np.array([self._update_state(memory[0], memory[3]) for memory in memory_batch])
        gameovers = np.array([memory[4] for memory in memory_batch])
        arg_q_max = self._main_dqn.get_best_action(self._session, next_states)
        q_values = self._target_dqn.get_q_values(self._session, next_states)
        double_q = q_values[range(len(memory_batch)), arg_q_max]
        target_q = rewards + (self._gamma * double_q * (1 - gameovers))
        loss, _ = self._main_dqn.train(
            self._session,
            states,
            target_q,
            actions
        )
        return loss

    def _update_networks(self):
        update_ops = []
        for i, op in enumerate(self._main_dqn_vars):
            update_ops.append(self._target_dqn_vars[i].assign(op.value()))
        for op in update_ops:
            self._session.run(op)

    def load(self, dir_path):
        self._saver.restore(self._session, dir_path + '/model')
        with open(os.path.join(dir_path, 'breakoutCheckpoint.json')) as checkpoint:
            self._steps = checkpoint['steps']

    def save(self, dir_path):
        self._saver.save(self._session, dir_path + '/model')
        with open(os.path.join(dir_path, 'breakoutCheckpoint.json'), 'w') as output_file:
            data = {
                "steps": self._steps
            }
            json.dump(data, output_file)


    @property
    def steps(self):
        return self._steps