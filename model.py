import tensorflow as tf
import numpy as np

class Model():
    def __init__(self, input_shape=[None, 84, 84, 4], num_outputs=4, learning_rate=0.0001):
        self._input_shape = input_shape
        self._num_outputs = num_outputs
        self._learning_rate = learning_rate
        self._define_model()

    def _define_model(self):
        self._input_layer = tf.placeholder(
            shape=self._input_shape,
            dtype=tf.float32,
            name="input"
        )
        self._scaled_inputs = self._input_layer / 255
        self._conv1 = tf.layers.conv2d(
            inputs=self._scaled_inputs,
            filters=32,
            kernel_size=[8, 8],
            strides=4,
            kernel_initializer=tf.variance_scaling_initializer(scale=2),
            padding="valid",
            activation=tf.nn.relu,
            use_bias=False,
            name="conv1"
        )
        self._conv2 = tf.layers.conv2d(
            inputs=self._conv1,
            filters=64,
            kernel_size=[4, 4],
            strides=2,
            kernel_initializer=tf.variance_scaling_initializer(scale=2),
            padding="valid",
            activation=tf.nn.relu,
            use_bias=False,
            name="conv2"
        )
        self._conv3 = tf.layers.conv2d(
            inputs=self._conv2,
            filters=64,
            kernel_size=[3, 3],
            strides=1,
            kernel_initializer=tf.variance_scaling_initializer(scale=2),
            padding="valid",
            activation=tf.nn.relu,
            use_bias=False,
            name="conv3"
        )
        self._conv4 = tf.layers.conv2d(
            inputs=self._conv3,
            filters=1024,
            kernel_size=[7, 7],
            strides=1,
            kernel_initializer=tf.variance_scaling_initializer(scale=2),
            padding="valid",
            activation=tf.nn.relu,
            use_bias=False,
            name="conv4"
        )

        self._value_stream, self._advantage_stream = tf.split(self._conv4, 2, 3)
        self._value_stream = tf.layers.flatten(self._value_stream)
        self._advantage_stream = tf.layers.flatten(self._advantage_stream)
        self._advantage = tf.layers.dense(
            inputs=self._advantage_stream,
            units=self._num_outputs,
            kernel_initializer=tf.variance_scaling_initializer(scale=2),
            name="advantage"
        )
        self._value = tf.layers.dense(
            inputs=self._value_stream,
            units=1,
            kernel_initializer=tf.variance_scaling_initializer(scale=2),
            name="value"
        )

        self._q_values = self._value + tf.subtract(
            self._advantage,
            tf.reduce_mean(
                self._advantage,
                axis=1,
                keep_dims=True
            )
        )
        self._best_action = tf.arg_max(self._q_values, 1)
        self._target_q = tf.placeholder(
            shape=[None],
            dtype=tf.float32
        )
        self._action = tf.placeholder(
            shape=[None],
            dtype=tf.int32
        )
        self._q = tf.reduce_sum(
            tf.multiply(
                self._q_values,
                tf.one_hot(
                    self._action,
                    self._num_outputs,
                    dtype=tf.float32
                )
            ),
            axis=1
        )
        self._loss = tf.reduce_mean(
            tf.losses.huber_loss(
                labels=self._target_q,
                predictions=self._q
            )
        )
        self._optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate)
        self._update = self._optimizer.minimize(self._loss)

    def _reshape_states(self, states):
        return np.reshape(
            states,
            (
                -1, 
                self._input_shape[1],
                self._input_shape[2],
                self._input_shape[3]
            )
        )

    def get_best_action(self, session, state):
        state = self._reshape_states(state)
        return session.run(self._best_action, feed_dict={self._input_layer: state})
    
    def get_q_values(self, session, states):
        states = self._reshape_states(states)
        return session.run(self._q_values, feed_dict={self._input_layer: states})
    
    def train(self, session, states, target_q, actions):
        states = self._reshape_states(states)
        return session.run(
            [
                self._loss, 
                self._update
            ], 
            feed_dict={
                self._input_layer: states,
                self._target_q: target_q,
                self._action: actions
            })