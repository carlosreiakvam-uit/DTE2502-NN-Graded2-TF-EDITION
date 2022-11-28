import json
import numpy as np
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense
from tensorflow.keras import Model
from agents.agent import Agent
from agents.agent import mean_huber_loss


class DeepQLearningAgent(Agent):

    def __init__(self, board_size=10, frames=4, buffer_size=10000, gamma=0.99, n_actions=3, use_target_net=True,
                 version=''):
        super().__init__(board_size, frames, buffer_size, gamma, n_actions, use_target_net, version)
        self._model = None
        self._target_net = None
        self.reset_models()

    def reset_models(self):
        self._model = self._agent_model()
        if self._use_target_net:  # true
            self._target_net = self._agent_model()
            self.update_target_net()

    def _prepare_input(self, board):
        # reshape board if it its dimension is equal to 3
        if (board.ndim == 3):
            board = board.reshape((1,) + self._input_shape)
        board = self._normalize_board(board.copy()) # cast to np.float32, somehow divided by 4
        return board.copy() # returns copy of board of same type (np.float32)

    # model = initialized tf model
    def _get_model_outputs(self, board, model=None):
        # to correct dimensions and normalize
        board = self._prepare_input(board)
        # the default model to use
        if model is None: # true
            model = self._model

        # TensorFlow method that predicts output based on batch
        # This one is important to get right in regards to a pytorch implementation
        model_outputs = model.predict_on_batch(board)
        return model_outputs

    def _normalize_board(self, board):
        # return copy of numpy board cast to type np.float32 divided by 4
        return board.astype(np.float32) / 4.0

    def move(self, board, legal_moves, value=None):
        # use the agent model to make the predictions
        model_outputs = self._get_model_outputs(board, self._model)
        return np.argmax(np.where(legal_moves == 1, model_outputs, -np.inf), axis=1)

    def _agent_model(self):
        # define the input layer, shape is dependent on the board size and frames
        with open('model_config/{:s}.json'.format(self._version), 'r') as f:
            m = json.loads(f.read())

        input_board = Input((self._board_size, self._board_size, self._n_frames,), name='input')
        x = input_board
        for layer in m['model']:
            l = m['model'][layer]
            if 'Conv2D' in layer:
                # add convolutional layer
                x = Conv2D(**l)(x)
            if 'Flatten' in layer:
                x = Flatten()(x)
            if 'Dense' in layer:
                x = Dense(**l)(x)
        out = Dense(self._n_actions, activation='linear', name='action_values')(x)
        model = Model(inputs=input_board, outputs=out)  # Keras model
        model.compile(optimizer=RMSprop(0.0005), loss=mean_huber_loss)

        return model

    def set_weights_trainable(self):
        """Set selected layers to non trainable and compile the model"""
        for layer in self._model.layers:
            layer.trainable = False
        # the last dense layers should be trainable
        for s in ['action_prev_dense', 'action_values']:
            self._model.get_layer(s).trainable = True
        self._model.compile(optimizer=self._model.optimizer,
                            loss=self._model.loss)

    def get_action_proba(self, board, values=None):
        model_outputs = self._get_model_outputs(board, self._model)
        # subtracting max and taking softmax does not change output
        # do this for numerical stability
        model_outputs = np.clip(model_outputs, -10, 10)
        model_outputs = model_outputs - model_outputs.max(axis=1).reshape((-1, 1))
        model_outputs = np.exp(model_outputs)
        model_outputs = model_outputs / model_outputs.sum(axis=1).reshape((-1, 1))
        return model_outputs

    def save_model(self, file_path='', iteration=None):
        if iteration is not None:
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        self._model.save_weights("{}/model_{:04d}.h5".format(file_path, iteration))
        if self._use_target_net:
            self._target_net.save_weights("{}/model_{:04d}_target.h5".format(file_path, iteration))

    def load_model(self, file_path='', iteration=None):
        if iteration is not None:
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        self._model.load_weights("{}/model_{:04d}.h5".format(file_path, iteration))
        if self._use_target_net:
            self._target_net.load_weights("{}/model_{:04d}_target.h5".format(file_path, iteration))
        # print("Couldn't locate models at {}, check provided path".format(file_path))

    def print_models(self):
        print('Training Model')
        print(self._model.summary())
        if self._use_target_net:
            print('Target Network')
            print(self._target_net.summary())

    # batch_size = 64, reward_clip = true
    def train_agent(self, batch_size=32, num_games=1, reward_clip=False):
        # where is self.buffer is a ReplayBuffer initialized from Agent
        # It is sampling batch_size number of examples from the buffer
        # getting batc_size number of: state, action, reward, next_state, done, legal_moves
        s, a, r, next_s, done, legal_moves = self._buffer.sample(batch_size)
        if reward_clip:  # true
            r = np.sign(r)  # returns batch_size numbers of: -1 if r<0, 0 if r== 0 or 1 if r > 0
            # this is in order to indicate if the reward is negative, positive or passive
            # essentially it is clipping the reward to discrete values

        # calculate the discounted reward, and then train accordingly
        # Decides wether or not to use target_net as current_model, even though target net is equal to current_model 🤔
        current_model = self._target_net if self._use_target_net else self._model

        # returns 64 x 4 outputs
        next_model_outputs = self._get_model_outputs(next_s, current_model)

        # our estimate of expexted future discounted reward
        # discounted_reward is a 64x4 tensor, a modified reward tensor
        # gamma is discount, set to 0.99
        # np.max determines max output of
        # np.where: numpy.where(condition, [x, y, ]/), where true yield x, otherwise yield y
        #   In other words if legal_moves == 1: yield next_model_outuout, else yiedl negative infinity
        # the where part is reshaped somehow
        # and finally multiplied with (1-done) which either yields 1 or 0
        #   meaning if not done, it makes the whole part 0
        discounted_reward = r + \
                            (self._gamma * np.max(
                                np.where(legal_moves == 1, next_model_outputs, -np.inf),
                                axis=1).reshape(-1, 1)) * (1 - done)

        # create the target variable, only the column with action has different value
        # This is another go at get_model_outout, only this time, only state is input
        target = self._get_model_outputs(s)

        # we bother only with the difference in reward estimate at the selected action
        # a is a 64x4 tensor
        # target is the model outputs, or labels if you will
        # discounted reward is what it sounds like
        target = (1 - a) * target + a * discounted_reward

        # fit
        # train_on_batch() is a tensorflow method (not to be mistaken for predict_on_batch)
        # states are normalized and used as input x
        # target is used for labels
        # the training provides, as indicated, the loss which we intend to minimize
        loss = self._model.train_on_batch(self._normalize_board(s), target)
        # loss = round(loss, 5)
        return loss

    def update_target_net(self):
        if self._use_target_net:  # true
            self._target_net.set_weights(self._model.get_weights())  # sets weights for _target_net from _model

    def compare_weights(self):
        for i in range(len(self._model.layers)):
            for j in range(len(self._model.layers[i].weights)):
                c = (self._model.layers[i].weights[j].numpy() == self._target_net.layers[i].weights[j].numpy()).all()
                print('Layer {:d} Weights {:d} Match : {:d}'.format(i, j, int(c)))

    def copy_weights_from_agent(self, agent_for_copy):
        assert isinstance(agent_for_copy, self), "Agent type is required for copy"

        self._model.set_weights(agent_for_copy._model.get_weights())
        self._target_net.set_weights(agent_for_copy._model_pred.get_weights())
