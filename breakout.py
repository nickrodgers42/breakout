import gym
import tensorflow as tf
import random

from replay_memory import ReplayMemory
from frame_processor import FrameProcessor
from model import Model

ENV_NAME = 'BreakoutDeterministic-v4'

class Breakout():
    def __init__(self, memory_size, memory_start_size, frame_buff_size, batch_size, 
        learning_rate, update_frequency, no_op_max, max_episode_length, 
        network_update_frequency, max_epsilon, min_epsilon, render=True):

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
        self._var_init = tf.global_variables_initializer()
        self._saver = tf.train.Saver()

        self._memory_start_size = memory_start_size
        self._update_frequency =  update_frequency
        self._no_op_max = no_op_max
        self._max_episode_length = max_episode_length
        self._network_update_frequency = network_update_frequency
        self._render = render

        self._max_epsilon = max_epsilon
        self._min_epsilon = min_epsilon

        self._steps = 0

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
                    self._steps += 1
                if gameover or self._memory.size >= self._memory_start_size:
                    break
            print(f'Filling Memory... Size: {self._memory.size}', end='\r')
        print('\nMemory Filled')

    def train(self):
        no_ops = random.randint(0, self._no_op_max)
        action = 1
        total_reward = 0
        episode_length = 0
        total_loss = 0
        last_lives = 5

        if self._memory.size < self._memory_start_size:
            self._fill_memory()

        # while True:
        #     if self._render:
        #         self._env.render()
        #     if no_ops > 0:


    
    def play(self):
        pass
    
    def save(self):
        pass

    def load(self):
        pass