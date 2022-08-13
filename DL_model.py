import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
#from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras import Model
from utils import *


def audio_to_text_rnn(input_dim, output_dim, rnn_layers=5, rnn_units = 128):
    my_input = Input(shape=(None,input_dim))
    x = tf.keras.layers.Reshape((-1,input_dim,1))(my_input)
    #x = Flatten()(my_input)
    x = Conv2D(32,(11,41),strides=[2,2],padding="same",use_bias=False,activation='relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(32,(11,21),strides=[1,2],padding='same',use_bias=False,activation='relu')(x)
    x = BatchNormalization()(x)

    #x = Flatten()(x)
    x = tf.keras.layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)
    #RNN Layers
    for i in range(1, rnn_layers + 1):
        recurrent = tf.keras.layers.GRU(
            units=rnn_units,
            activation="tanh",
            recurrent_activation="sigmoid",
            use_bias=True,
            return_sequences=True,
            reset_after=True
        )
        x = tf.keras.layers.Bidirectional(
            recurrent, name=f"bidirectional_{i}", merge_mode="concat"
        )(x)
        if i < rnn_layers:
            x = Dropout(rate=0.5)(x)
    # Dense layer
    x = Dense(units=rnn_units*2,activation='relu')(x)
    x = Dropout(0.5)(x)

    #classification layer
    output = Dense(units=output_dim+1,activation='softmax')(x)
    model = Model(my_input,output,name='SPEECH_RNN')
    return model


if __name__ == '__main__':
    input_dim = (384//2)+1
    output_dim = 35
    model = audio_to_text_rnn(input_dim, output_dim, rnn_layers=5, rnn_units = 128)
    model.summary()
