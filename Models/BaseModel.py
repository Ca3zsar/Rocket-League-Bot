import tensorflow.keras.models
from tensorflow.keras import optimizers
from tensorflow.python.keras.layers import Dense, InputLayer


class BaseModel(tensorflow.keras.models.Sequential):
    def __init__(self, state_shape, action_number):
        super(BaseModel, self).__init__()

        self.state_shape = state_shape
        self.action_number = action_number

        self.initialize_model()

    def initialize_model(self):
        self.add(InputLayer(input_shape=(self.state_shape,)))
        self.add(Dense(128, activation='relu', kernel_initializer='glorot_uniform'))
        self.add(Dense(128, activation='relu', kernel_initializer='glorot_uniform'))
        self.add(Dense(self.action_number, activation='linear'))

        self.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
                     loss='mse',
                     metrics=['accuracy'])
