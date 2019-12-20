import tensorflow as tf
from breakout import Breakout

MAX_STEPS = 30000000
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
GAMMA = 0.99


def train_breakout():
    with tf.Session() as sess:
        breakout = Breakout(
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
            min_epsilon=MIN_EPSILON,
            gamma=GAMMA
        )
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        sess.run(init)
        episode_count = 0
        while breakout.steps < MAX_STEPS:
            breakout.train()
            if episode_count % 10 == 0:
                print(f'\nEpisode {episode_count}')
            if episode_count % 500 == 0 and episode_count > 0:
                saver.save(sess, './nets/net' + str(episode_count) + '/model')
            episode_count += 1

if __name__ == '__main__':
    train_breakout()
