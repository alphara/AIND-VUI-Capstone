from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input,
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM,
    MaxPooling1D)

def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True,
                 implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    # DONE: Add batch normalization
    bn_rnn = BatchNormalization(name='bn')(simp_rnn)
    # DONE: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim, name='dense'), name='time_dense')(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size,
                     strides=conv_stride,
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = SimpleRNN(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    # DONE: Add batch normalization
    bn_rnn = BatchNormalization(name='bn')(simp_rnn)
    # DONE: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim, name='dense'), name='time_dense')(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))

    # DONE: Add recurrent layers, each with batch normalization
    recur = input_data
    for i in range(recur_layers):
        recur = GRU(units, activation='relu', return_sequences=True,
                    implementation=2, name='recur_{}'.format(i))(recur)
        recur = BatchNormalization(name='recur_bn_{}'.format(i))(recur)

    # DONE: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim, name='dense'), name='time_dense')(recur)

    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))

    # DONE: Add bidirectional recurrent layer
    bidir_rnn = Bidirectional(GRU(units, return_sequences=True,
                                  activation='relu', implementation=2,
                                  name='rnn'),
                              merge_mode='concat', name='bidir_rnn')(input_data)

    # DONE: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim, name='dense'),
                                 name='time_dense')(bidir_rnn)

    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def deeper_cnn_bidir_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, bidir_rnn_layers, rnn_layers, output_dim=29):
    """ Convolutional and bidirectional recurrent neural network for speech
    """
    input_data = Input(name='the_input', shape=(None, input_dim))
    conv_1d = Conv1D(filters, kernel_size,
                     strides=conv_stride,
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)

    recur = bn_cnn
    for i in range(bidir_rnn_layers):
        recur = Bidirectional(SimpleRNN(units,
                                        activation='relu',
                                        return_sequences=True,
                                        implementation=2,
                                        name='rnn_{}'.format(i)),
                              merge_mode='concat',
                              name='bd_rnn_{}'.format(i))(recur)
        recur = BatchNormalization(name='bn_{}'.format(i))(recur)
    for i in range(rnn_layers):
        recur = SimpleRNN(units,
                          activation='relu',
                          return_sequences=True,
                          implementation=2,
                          name='s_rnn_{}'.format(i))(recur)
        recur = BatchNormalization(name='s_bn_{}'.format(i))(recur)

    time_dense = TimeDistributed(Dense(output_dim, name='dense'), name='time_dense')(recur)
    y_pred = Activation('softmax', name='softmax')(time_dense)
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def dilated_cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, dilation_rate, units, output_dim=29):
    """ Dilated convolutional and recurrent network for speech
    """
    input_data = Input(name='the_input', shape=(None, input_dim))

    conv_1d = Conv1D(filters, kernel_size,
                     strides=conv_stride,
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    dilated_conv_1d = Conv1D(filters, kernel_size,
                     strides=1,
                     padding=conv_border_mode,
                     activation='relu',
                     dilation_rate=dilation_rate,
                     name='dilated_conv1d')(conv_1d)

    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(dilated_conv_1d)
    # Add a recurrent layer
    simp_rnn = SimpleRNN(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    # DONE: Add batch normalization
    bn_rnn = BatchNormalization(name='bn')(simp_rnn)
    # DONE: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim, name='dense'), name='time_dense')(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
      cnn_output_length(x, kernel_size, conv_border_mode, conv_stride),
      kernel_size, conv_border_mode, 1, dilation=dilation_rate)

    print(model.summary())
    return model

def cnn_rnn_dropout_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, dropout, recurrent_dropout, output_dim=29):
    """ Build a recurrent + convolutional network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size,
                     strides=conv_stride,
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = SimpleRNN(units, activation='relu',
        return_sequences=True, implementation=2,
        dropout=dropout, recurrent_dropout=recurrent_dropout,
        name='rnn')(bn_cnn)
    # DONE: Add batch normalization
    bn_rnn = BatchNormalization(name='bn')(simp_rnn)
    # DONE: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim, name='dense'), name='time_dense')(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def dilated_deeper_cnn_bidir_rnn_dropout_model(input_dim, filters,
    kernel_size, conv_stride, conv_border_mode, dilation_rate, units,
    bidir_rnn_layers, dropout, recurrent_dropout, output_dim=29):
    """ Convolutional and bidirectional recurrent neural network for speech
    """
    input_data = Input(name='the_input', shape=(None, input_dim))
    conv_1d = Conv1D(filters, kernel_size,
                     strides=conv_stride,
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    dilated_conv_1d = Conv1D(filters, kernel_size,
                     strides=1,
                     padding=conv_border_mode,
                     activation='relu',
                     dilation_rate=dilation_rate,
                     name='dilated_conv1d')(conv_1d)

    bn_cnn = BatchNormalization(name='bn_conv_1d')(dilated_conv_1d)

    recur = bn_cnn
    for i in range(bidir_rnn_layers):
        recur = Bidirectional(SimpleRNN(units,
                                        activation='relu',
                                        return_sequences=True,
                                        implementation=2,
                                        dropout=dropout,
                                        recurrent_dropout=recurrent_dropout,
                                        name='rnn_{}'.format(i)),
                              merge_mode='concat',
                              name='bd_rnn_{}'.format(i))(recur)
        recur = BatchNormalization(name='bn_{}'.format(i))(recur)

    time_dense = TimeDistributed(Dense(output_dim, name='dense'), name='time_dense')(recur)
    y_pred = Activation('softmax', name='softmax')(time_dense)
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
      cnn_output_length(x, kernel_size, conv_border_mode, conv_stride),
      kernel_size, conv_border_mode, 1, dilation=dilation_rate)
    print(model.summary())
    return model

def final_model():
    """ Build a deep network for speech
    """
    return dilated_deeper_cnn_bidir_rnn_dropout_model(input_dim=161, # change to 13 if you would like to use MFCC features
                                      filters=250,
                                      kernel_size=11,
                                      conv_stride=2,
                                      conv_border_mode='valid',
                                      dilation_rate=2,
                                      units=300,
                                      bidir_rnn_layers=1,
                                      dropout=0.1,
                                      recurrent_dropout=0.05
                                     )
