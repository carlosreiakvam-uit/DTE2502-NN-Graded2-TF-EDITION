from replay_buffer import ReplayBuffer, ReplayBufferNumpy
import numpy as np
import pickle
import tensorflow as tf


class Agent:
    def __init__(self, board_size=10, frames=2, buffer_size=10000,
                 gamma=0.99, n_actions=3, use_target_net=True,
                 version=''):
        self._board_size = board_size
        self._n_frames = frames
        self._buffer_size = buffer_size
        self._n_actions = n_actions
        self._gamma = gamma
        self._use_target_net = use_target_net
        self._input_shape = (self._board_size, self._board_size, self._n_frames)
        # reset buffer also initializes the buffer
        self.reset_buffer()
        self._board_grid = np.arange(0, self._board_size ** 2) \
            .reshape(self._board_size, -1)
        self._version = version

    def get_gamma(self) -> float:
        return self._gamma

    def reset_buffer(self, buffer_size=None):
        if (buffer_size is not None):
            self._buffer_size = buffer_size
        self._buffer = ReplayBufferNumpy(self._buffer_size, self._board_size,
                                         self._n_frames, self._n_actions)

    def get_buffer_size(self):
        return self._buffer.get_current_size()

    def add_to_buffer(self, board, action, reward, next_board, done, legal_moves):
        self._buffer.add_to_buffer(board, action, reward, next_board,
                                   done, legal_moves)

    def save_buffer(self, file_path='', iteration=None):
        if (iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        with open("{}/buffer_{:04d}".format(file_path, iteration), 'wb') as f:
            pickle.dump(self._buffer, f)

    def load_buffer(self, file_path='', iteration=None):
        if (iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        with open("{}/buffer_{:04d}".format(file_path, iteration), 'rb') as f:
            self._buffer = pickle.load(f)

    def _point_to_row_col(self, point):
        return (point // self._board_size, point % self._board_size)

    def _row_col_to_point(self, row, col):
        return row * self._board_size + col


def huber_loss(y_true, y_pred, delta=1):
    error = (y_true - y_pred)
    quad_error = 0.5 * tf.math.square(error)
    lin_error = delta * (tf.math.abs(error) - 0.5 * delta)
    # quadratic error, linear error
    return tf.where(tf.math.abs(error) < delta, quad_error, lin_error)


def mean_huber_loss(y_true, y_pred, delta=1):
    return tf.reduce_mean(huber_loss(y_true, y_pred, delta))
