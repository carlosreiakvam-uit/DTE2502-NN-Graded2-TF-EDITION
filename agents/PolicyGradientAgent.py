import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from agents.DeepQLearningAgent import DeepQLearningAgent


class PolicyGradientAgent(DeepQLearningAgent):

    def __init__(self, board_size=10, frames=4, buffer_size=10000, gamma=0.99, n_actions=3, use_target_net=False,
                 version=''):
        super().__init__(board_size, frames, buffer_size, gamma, n_actions, use_target_net, version)
        self._actor_optimizer = tf.keras.optimizer.Adam(1e-6)

    def _agent_model(self):
        input_board = Input((self._board_size, self._board_size, self._n_frames,))
        x = Conv2D(16, (4, 4), activation='relu', data_format='channels_last', kernel_regularizer=l2(0.01))(input_board)
        x = Conv2D(32, (4, 4), activation='relu', data_format='channels_last', kernel_regularizer=l2(0.01))(x)
        x = Flatten()(x)
        x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)
        out = Dense(self._n_actions, activation='linear', name='action_logits', kernel_regularizer=l2(0.01))(x)

        model = Model(inputs=input_board, outputs=out)
        # do not compile the model here, but rather use the outputs separately
        # in a training function to create any custom loss function
        # model.compile(optimizer = RMSprop(0.0005), loss = 'mean_squared_error')
        return model

    def train_agent(self, batch_size=32, beta=0.1, normalize_rewards=False,
                    num_games=1, reward_clip=False):
        # in policy gradient, only complete episodes are used for training
        s, a, r, _, _, _ = self._buffer.sample(self._buffer.get_current_size())
        # unlike DQN, the discounted reward is not estimated but true one
        # we have defined custom policy graident loss function above
        # use that to train to agent model
        # normzlize the rewards for training stability
        if (normalize_rewards):
            r = (r - np.mean(r)) / (np.std(r) + 1e-8)
        target = np.multiply(a, r)
        loss = actor_loss_update(self._prepare_input(s), target, self._model,
                                 self._actor_optimizer, beta=beta, num_games=num_games)
        return loss[0] if len(loss) == 1 else loss
