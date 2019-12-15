import tensorflow as tf
from breakout import Breakout

MEMORY_SIZE = 1000000
MEMORY_START_SIZE = 50000
FRAME_BUFF = 4
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
UPDATE_FREQUENCY = 4
NO_OP_MAX = 15
MAX_EPISODE_LENGTH = 18000
NETWORK_UPDATE_FREQUENCY = 10000
MAX_EPSILON = 1
MIN_EPSILON = 0.1


def train_breakout():
    with tf.Session() as sess:
        breakout = Breakout(
            memory_size=MEMORY_SIZE,
            memory_start_size=MEMORY_START_SIZE,
            frame_buff_size=FRAME_BUFF,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            update_frequency=UPDATE_FREQUENCY,
            no_op_max=NO_OP_MAX,
            max_episode_length=MAX_EPISODE_LENGTH,
            network_update_frequency=NETWORK_UPDATE_FREQUENCY
        )
        breakout.train()

if __name__ == '__main__':
    train_breakout()
