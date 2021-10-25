# Hannah Kerner
# July 6, 2020

import keras
from keras.layers import BatchNormalization
from keras.models import Model
from keras.layers import Dense
from keras.layers import concatenate
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Conv1D, Conv2D
from keras.layers import MaxPool2D
from keras.layers import Dropout
from keras.layers import Reshape
from keras.layers import Flatten

N_CLASSES = 3

def lstm_cnn_delta(n_units_lstm=[8, 16, 16, 16, 16], kernels_lstm=[3], 
                   n_units_cnn=[64, 64], kernels_cnn=[3, 3],
                   fc_units=64,
                   n_bands=11, n_timesteps=3, patchdim=5):
    ######## LSTM branch ########
    input_shape = (n_timesteps, n_bands)
    lstm_inputs = Input(shape=input_shape)

    n_layers = len(n_units_lstm)

    # First process the input through a 1D conv
    # TODO: test replacing Conv1D with linear layer
    #x_lstm = Conv1D(filters=n_units_lstm[0], kernel_size=kernels_lstm[0], padding='same', activation='relu')(lstm_inputs)
    x_lstm = Dense(units=n_bands*2)(lstm_inputs)
    x_lstm = BatchNormalization()(x_lstm)

    # TODO: add dense layers/skip connections
    # Add each layer to the graph
    for l in range(1, n_layers):
        x_lstm = LSTM(n_units_lstm[l], input_shape=x_lstm.shape, return_sequences=True, activation='relu')(x_lstm)
        x_lstm = BatchNormalization()(x_lstm)
        x_lstm = Dropout(0.3)(x_lstm)

    # Flatten the LSTM outputs
    flat_lstm = Flatten()(x_lstm)

    # Build the LSTM branch
    lstm = Model(inputs=lstm_inputs, outputs=flat_lstm)

    ######## CNN branch ########
    cnn_inputs = Input(shape=(patchdim, patchdim, n_timesteps*n_bands))
    
    # Add each layer to the graph
    for l in range(len(n_units_cnn)):
        if l == 0:
            x_cnn = Conv2D(filters=n_units_cnn[l], kernel_size=kernels_cnn[l], padding='same', activation='relu')(cnn_inputs)
        else:
            x_cnn = Conv2D(filters=n_units_cnn[l], kernel_size=kernels_cnn[l], padding='same', activation='relu')(x_cnn)
        x_cnn = BatchNormalization()(x_cnn)
        # TODO: test with dropout?

    x_cnn = MaxPool2D()(x_cnn)

    flat_cnn = Flatten()(x_cnn)

    # Build the CNN branch
    cnn = Model(inputs=cnn_inputs, outputs=flat_cnn)

    ########## Scalar timeline branch ###########
    delta = Input(shape=(1,))

    combined = concatenate([lstm.output, cnn.output, delta])

    fc = Dense(fc_units, activation='relu')(combined)
    fc = Dropout(0.3)(fc)

    # output layer -> corn, soybeans, other 
    preds = Dense(N_CLASSES, activation='softmax')(fc)

    # Build the model
    model = Model(inputs=[lstm.input, cnn.input, delta], outputs=preds)
    return model

def lstm_cnn(n_units_lstm=[8, 16, 16, 16, 16], kernels_lstm=[3], 
               n_units_cnn=[64, 64], kernels_cnn=[3, 3],
               fc_units=64,
               n_bands=11, n_timesteps=3, patchdim=5):
    ######## LSTM branch ########
    input_shape = (n_timesteps, n_bands)
    lstm_inputs = Input(shape=input_shape)

    n_layers = len(n_units_lstm)

    # First process the input through a 1D conv
    # TODO: test replacing Conv1D with linear layer
    #x_lstm = Conv1D(filters=n_units_lstm[0], kernel_size=kernels_lstm[0], padding='same', activation='relu')(lstm_inputs)
    x_lstm = Dense(units=n_bands*2)(lstm_inputs)
    x_lstm = BatchNormalization()(x_lstm)

    # TODO: add dense layers/skip connections
    # Add each layer to the graph
    for l in range(1, n_layers):
        x_lstm = LSTM(n_units_lstm[l], input_shape=x_lstm.shape, return_sequences=True, activation='relu')(x_lstm)
        x_lstm = BatchNormalization()(x_lstm)
        x_lstm = Dropout(0.3)(x_lstm)

    # Flatten the LSTM outputs
    flat_lstm = Flatten()(x_lstm)

    # Build the LSTM branch
    lstm = Model(inputs=lstm_inputs, outputs=flat_lstm)

    ######## CNN branch ########
    cnn_inputs = Input(shape=(patchdim, patchdim, n_timesteps*n_bands))
    
    # Add each layer to the graph
    for l in range(len(n_units_cnn)):
        if l == 0:
            x_cnn = Conv2D(filters=n_units_cnn[l], kernel_size=kernels_cnn[l], padding='same', activation='relu')(cnn_inputs)
        else:
            x_cnn = Conv2D(filters=n_units_cnn[l], kernel_size=kernels_cnn[l], padding='same', activation='relu')(x_cnn)
        x_cnn = BatchNormalization()(x_cnn)
        # TODO: test with dropout?

    x_cnn = MaxPool2D()(x_cnn)

    flat_cnn = Flatten()(x_cnn)

    # Build the CNN branch
    cnn = Model(inputs=cnn_inputs, outputs=flat_cnn)

    combined = concatenate([lstm.output, cnn.output])

    fc = Dense(fc_units, activation='relu')(combined)
    fc = Dropout(0.3)(fc)

    # output layer -> corn, soybeans, other 
    preds = Dense(N_CLASSES, activation='softmax')(fc)

    # Build the model
    model = Model(inputs=[lstm.input, cnn.input], outputs=preds)
    return model

def cnn(n_units_cnn=[64, 64], kernels_cnn=[3, 3],
        fc_units=64,
        n_bands=11, n_timesteps=3, patchdim=5):

    ######## CNN branch ########
    cnn_inputs = Input(shape=(patchdim, patchdim, n_timesteps*n_bands))
    
    # Add each layer to the graph
    for l in range(len(n_units_cnn)):
        if l == 0:
            x_cnn = Conv2D(filters=n_units_cnn[l], kernel_size=kernels_cnn[l], 
                padding='same', activation='relu')(cnn_inputs)
        else:
            x_cnn = Conv2D(filters=n_units_cnn[l], kernel_size=kernels_cnn[l], 
                padding='same', activation='relu')(x_cnn)
        x_cnn = BatchNormalization()(x_cnn)
    x_cnn = MaxPool2D()(x_cnn)

    flat_cnn = Flatten()(x_cnn)

    fc = Dense(fc_units, activation='relu')(flat_cnn)
    fc = Dropout(0.3)(fc)

    # output layer -> corn, soybeans, other 
    preds = Dense(N_CLASSES, activation='softmax')(fc)

    # Build the model
    model = Model(inputs=[cnn_inputs], outputs=preds)
    return model

def lstm(n_units_lstm=[8, 16, 16, 16, 16], kernels_lstm=[3], 
           fc_units=64,
           n_bands=11, n_timesteps=3):
    ######## LSTM branch ########
    input_shape = (n_timesteps, n_bands)
    lstm_inputs = Input(shape=input_shape)

    n_layers = len(n_units_lstm)

    # First process the input through a 1D conv
    # TODO: test replacing Conv1D with linear layer
    #x_lstm = Conv1D(filters=n_units_lstm[0], kernel_size=kernels_lstm[0], padding='same', activation='relu')(lstm_inputs)
    x_lstm = Dense(units=n_bands*2)(lstm_inputs)
    x_lstm = BatchNormalization()(x_lstm)

    # TODO: add dense layers/skip connections
    # Add each layer to the graph
    for l in range(1, n_layers):
        x_lstm = LSTM(n_units_lstm[l], input_shape=x_lstm.shape, return_sequences=True, activation='relu')(x_lstm)
        x_lstm = BatchNormalization()(x_lstm)
        x_lstm = Dropout(0.3)(x_lstm)

    # Flatten the LSTM outputs
    flat_lstm = Flatten()(x_lstm)

    fc = Dense(fc_units, activation='relu')(flat_lstm)
    fc = Dropout(0.3)(fc)

    # output layer -> corn, soybeans, other 
    preds = Dense(N_CLASSES, activation='softmax')(fc)

    # Build the model
    model = Model(inputs=[lstm_inputs], outputs=preds)
    return model