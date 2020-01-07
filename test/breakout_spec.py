import unittest
import tensorflow as tf

import sys
sys.path.insert(1, '../')
from breakout import Breakout

class TestBreakout(unittest.TestCase):
    def setUp(self):
        MEMORY_SIZE = 600
        MEMORY_START_SIZE = 500
        FRAME_BUFF = 4
        BATCH_SIZE = 32
        LEARNING_RATE = 0.0001
        UPDATE_FREQUENCY = 4
        NO_OP_MAX = 15
        MAX_EPISODE_LENGTH = 18000
        NETWORK_UPDATE_FREQUENCY = 10000
        MAX_EPSILON = 1
        MIN_EPSILON = 0.1

        with tf.Session() as sess:
            self.breakout = Breakout(
                session=sess,
                memory_size=MEMORY_SIZE,
                memory_start_size=MEMORY_START_SIZE,
                frame_buff_size=FRAME_BUFF,
                batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATE,
                update_frequency=UPDATE_FREQUENCY,
                no_op_max=NO_OP_MAX,
                max_episode_length=MAX_EPISODE_LENGTH,
                network_update_frequency=NETWORK_UPDATE_FREQUENCY,
                max_epsilon=MAX_EPSILON,
                min_epsilon=MIN_EPSILON
            )