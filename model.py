from keras import backend as K, Input
from keras.layers import Dense, Conv1D
from keras.layers import GRU
from keras.layers import Lambda, Dot, Activation, Concatenate, Layer
from keras.models import Sequential
from keras.optimizers.legacy import RMSprop

from utils import CLASSES


class MultiHeadAttention(Layer):
    def __init__(self, units=128, **kwargs):
        print("[INFO] Building HopField Layer")
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        with K.name_scope("hopfield"):
            self.attention_score_vec = Dense(
                input_dim, use_bias=False, name="attention_score_vec"
            )
            self.h_t = Lambda(
                lambda x: x[:, -1, :],
                output_shape=(input_dim,),
                name="last_hidden_state",
            )
            self.attention_score = Dot(axes=[1, 2], name="attention_score")
            self.attention_weight = Activation("softmax", name="attention_weight")
            self.context_vector = Dot(axes=[1, 1], name="context_vector")
            self.attention_output = Concatenate(name="attention_output")
            self.attention_vector = Dense(
                self.units, use_bias=False, activation="tanh", name="attention_vector"
            )
            super(MultiHeadAttention, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.units

    def __call__(self, inputs, training=None, **kwargs):
        return super(MultiHeadAttention, self).__call__(inputs, training, **kwargs)

    def call(self, inputs, training=None, **kwargs):
        score_first_part = self.attention_score_vec(inputs)
        h_t = self.h_t(inputs)
        score = self.attention_score([h_t, score_first_part])
        attention_weights = self.attention_weight(score)
        context_vector = self.context_vector([inputs, attention_weights])
        pre_activation = self.attention_output([context_vector, h_t])
        attention_vector = self.attention_vector(pre_activation)
        return attention_vector

    def get_config(self):
        config = super(MultiHeadAttention, self).get_config()
        config.update({"units": self.units})
        return config


def crnn(shape):
    model_ = Sequential(name="crnn")
    model_.add(Input(shape=shape))
    model_.add(Conv1D(filters=128, kernel_size=3, padding="same", activation="relu"))
    model_.add(GRU(64, return_sequences=True))
    model_.add(MultiHeadAttention(32))
    model_.add(Dense(16, activation="relu"))
    model_.add(Dense(len(CLASSES), activation="softmax"))

    optimizer = RMSprop(lr=0.001)
    model_.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model_