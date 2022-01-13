from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers, initializers
from tensorflow.keras.layers import Dense, InputLayer, LeakyReLU
from tensorflow.python.keras.initializers.initializers_v2 import HeUniform, HeNormal, RandomNormal, TruncatedNormal
class BaseModel:
    def __init__(self, state_shape, action_number):
        self.state_shape = state_shape
        self.action_number = action_number
        self.model = Sequential()
        self.initialize_model()

    def initialize_model(self):
        self.model.add(InputLayer(input_shape=self.state_shape))
        self.model.add(Dense(64, activation='elu',kernel_initializer=HeUniform(),bias_initializer=initializers.Constant(0.1)))
        self.model.add(Dense(128, activation='elu',kernel_initializer=HeUniform(),bias_initializer=initializers.Constant(0.1)))
        self.model.add(LeakyReLU(0.25))
        self.model.add(Dense(self.action_number, activation='elu',kernel_initializer=HeUniform(),bias_initializer=initializers.Constant(0.1)))

        self.model.compile(optimizer=optimizers.SGD(learning_rate=0.0005),
                           loss='mse',
                           metrics=['accuracy', 'mse']
                           )
