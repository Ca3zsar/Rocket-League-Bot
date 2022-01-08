import tensorflow.keras.models
from tensorflow.keras import optimizers
from tensorflow.python.keras.layers import Dense, InputLayer, Lambda

from Models.BaseModel import BaseModel


class Actor(BaseModel):
    def __init__(self, state_shape, action_number):
        super(BaseModel, self).__init__(state_shape, action_number)
        self.initialize_model()


    def initialize_model(self):
        state = InputLayer(batch_shape=(None, self.state_size))
        actor_input = Dense(self.hidden1, input_dim=self.state_size, activation='relu')(state)
        actor_hidden = Dense(self.hidden2, activation='relu')(actor_input)
        mu_0 = Dense(self.action_size, activation='tanh')(actor_hidden)
        sigma_0 = Dense(self.action_size, activation='softplus')(actor_hidden)

        mu = Lambda(lambda x: x * 2)(mu_0)
        sigma = Lambda(lambda x: x + 0.0001)(sigma_0)

        # actor._make_predict_function()
        # actor.summary()
        # return actor

