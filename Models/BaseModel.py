from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers, initializers
from tensorflow.keras.layers import Dense, InputLayer, LeakyReLU
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.initializers.initializers_v2 import HeUniform, HeNormal, RandomNormal, TruncatedNormal
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.losses import CategoricalCrossentropy

class BaseModel:
    def __init__(self, state_shape, action_number):
        self.state_shape = state_shape
        self.action_number = action_number
        self.model = Sequential()
        self.initialize_model()

    def initialize_model(self):
        self.model.add(InputLayer(input_shape=self.state_shape))
        self.model.add(Dense(64, activation='elu',kernel_initializer=HeUniform(),bias_initializer=initializers.Constant(0.1)))
        self.model.add(Dense(128,activation='elu',kernel_initializer=HeUniform(),bias_initializer=initializers.Constant(0.1)))
        self.model.add(Dense(self.action_number, activation='linear',kernel_initializer=HeUniform(),bias_initializer=initializers.Constant(0.1)))

        self.model.compile(optimizer=optimizers.SGD(learning_rate=0.005),
                           loss='mse',
                           metrics=['accuracy']
                           )
