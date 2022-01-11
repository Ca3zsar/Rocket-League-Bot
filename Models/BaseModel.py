from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, InputLayer

class BaseModel:
    def __init__(self, state_shape, action_number):
        self.state_shape = state_shape
        self.action_number = action_number
        self.model = Sequential()
        self.initialize_model()

    def initialize_model(self):
        self.model.add(InputLayer(input_shape=self.state_shape))
        self.model.add(Dense(128, activation='relu', kernel_initializer='glorot_uniform'))
        self.model.add(Dense(128, activation='relu', kernel_initializer='glorot_uniform'))
        self.model.add(Dense(self.action_number, activation='relu'))

        self.model.compile(optimizer=optimizers.SGD(learning_rate=0.0001),
                           loss='mse',
                           metrics=['accuracy'])
