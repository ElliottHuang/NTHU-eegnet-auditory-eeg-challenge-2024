"""2 mismatched segments dilation model."""
import tensorflow as tf

import logging
import time

import numpy as np

import util2

def positional_encoding(length, depth):
  depth = depth/2

  positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
  depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

  angle_rates = 1 / (10000**depths)         # (1, depth)
  angle_rads = positions * angle_rates      # (pos, depth)

  pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)],
      axis=-1) 

  return tf.cast(pos_encoding, dtype=tf.float32)

class BaseAttention(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
    self.layernorm = tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()

class CrossAttention(BaseAttention):
  def call(self, x, context):
    attn_output, attn_scores = self.mha(
        query=x,
        key=context,
        value=context,
        return_attention_scores=True)

    # Cache the attention scores for plotting later.
    self.last_attn_scores = attn_scores

    x = self.add([x, attn_output])
    x = self.layernorm(x)

    return x

class GlobalSelfAttention(BaseAttention):
  def call(self, x):
    attn_output = self.mha(
        query=x,
        value=x,
        key=x)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x

class FeedForward(tf.keras.layers.Layer):
  def __init__(self, d_model, dff, dropout_rate=0.1):
    super().__init__()
    self.seq = tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),
      tf.keras.layers.Dense(d_model),
      tf.keras.layers.Dropout(dropout_rate)
    ])
    self.add = tf.keras.layers.Add()
    self.layer_norm = tf.keras.layers.LayerNormalization()

  def call(self, x):
    x = self.add([x, self.seq(x)])
    x = self.layer_norm(x) 
    return x

class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self,*, d_model, num_heads, dff, dropout_rate=0.1):
    super().__init__()

    self.self_attention = GlobalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.ffn = FeedForward(d_model, dff)

  def call(self, x):
    x = self.self_attention(x)
    x = self.ffn(x)
    return x

class Encoder(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, d_model, num_heads,
               dff, length, dropout_rate=0.0):
    super().__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.pos_encoding = positional_encoding(length,d_model)

    self.enc_layers = [
        EncoderLayer(d_model=d_model,
                     num_heads=num_heads,
                     dff=dff,
                     dropout_rate=dropout_rate)
        for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(dropout_rate)

  def call(self, x):
    # `x` is token-IDs shape: (batch, seq_len)
    x = x + self.pos_encoding  # Shape `(batch_size, seq_len, d_model)`.

    # Add dropout.
    x = self.dropout(x)

    for i in range(self.num_layers):
      x = self.enc_layers[i](x)

    return x  # Shape `(batch_size, seq_len, d_model)`.

def lstm_mel(shape_eeg, shape_spch, units_lstm=32, filters_cnn_eeg=16, filters_cnn_env=16,
                            units_hidden=128,
                            stride_temporal=3, kerSize_temporal=9, spatial_filters_eeg=32,
                            spatial_filters_mel=8, fun_act='tanh',num_mismatched_segments=2,compile=True):
    """
    Return an LSTM based model where batch normalization is applied to input of each layer.

    :param shape_eeg: a numpy array, shape of EEG signal (time, channel)
    :param shape_spch: a numpy array, shape of speech signal (time, feature_dim)
    :param units_lstm: an int, number of units in LSTM
    :param filters_cnn_eeg: an int, number of CNN filters applied on EEG
    :param filters_cnn_env: an int, number of CNN filters applied on envelope
    :param units_hidden: an int, number of units in the first time_distributed layer
    :param stride_temporal: an int, amount of stride in the temporal direction
    :param kerSize_temporal: an int, size of CNN filter kernel in the temporal direction
    :param fun_act: activation function used in layers
    :return: LSTM-based model
    """

    ############
    input_eeg = tf.keras.layers.Input(shape=shape_eeg)
    input_spch = [tf.keras.layers.Input(shape=shape_spch) for _ in range(num_mismatched_segments+1)]
    all_inputs = [input_eeg]
    all_inputs.extend(input_spch)
    
    ############
    #### upper part of network dealing with EEG.

    layer_exp1 = tf.keras.layers.Lambda(lambda x: tf.keras.backend.expand_dims(x, axis=3))
    eeg_proj = input_eeg

    # layer
    output_eeg = tf.keras.layers.BatchNormalization()(eeg_proj)  # batch normalization
    # output_eeg = tf.keras.layers.Conv1D(spatial_filters_eeg, kernel_size=1)(output_eeg)
    output_eeg = tf.keras.layers.Conv1D(spatial_filters_eeg, kernel_size=1)(output_eeg)
    # output_eeg = tf.keras.layers.Dropout(0.3)(output_eeg)  # dropout
    # layer
    output_eeg = tf.keras.layers.BatchNormalization()(output_eeg)
    output_eeg = layer_exp1(output_eeg)
    # output_eeg = tf.keras.layers.Convolution2D(filters_cnn_eeg, (kerSize_temporal, 1),
                                            #    strides=(stride_temporal, 1), activation="relu")(output_eeg)
    output_eeg = tf.keras.layers.Convolution2D(
        filters_cnn_eeg,
    (kerSize_temporal, 1),
    strides=(stride_temporal, 1),
    activation="relu"  # Adjust the regularization strength
    )(output_eeg)
    # output_eeg = tf.keras.layers.Dropout(0.3)(output_eeg)  # dropout

    # layer
    layer_permute = tf.keras.layers.Permute((1, 3, 2))
    output_eeg = layer_permute(output_eeg)

    layer_reshape = tf.keras.layers.Reshape((tf.keras.backend.int_shape(output_eeg)[1],
                                             tf.keras.backend.int_shape(output_eeg)[2] *
                                             tf.keras.backend.int_shape(output_eeg)[3]))
    output_eeg = layer_reshape(output_eeg)

    layer2_timeDis = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units_hidden, activation=fun_act))
    output_eeg = layer2_timeDis(output_eeg)

    # layer
    output_eeg = tf.keras.layers.BatchNormalization()(output_eeg)
    layer3_timeDis = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units_lstm, activation=fun_act))
    output_eeg = layer3_timeDis(output_eeg)

    ##############
    #### Bottom part of the network dealing with Speech.

    # spch_proj = [layer(spch_proji) for spch_proji in spch_proj]
    spch_proj = input_spch
   
    # layer
    BN_layer = tf.keras.layers.BatchNormalization()
    spch_proj = [BN_layer(spch_proji) for spch_proji in spch_proj]
   
    env_spatial_layer = tf.keras.layers.Conv1D(spatial_filters_mel, kernel_size=1)
    spch_proj = [env_spatial_layer(spch_proji) for spch_proji in spch_proj]
    
    # layer
    BN_layer1 = tf.keras.layers.BatchNormalization()
    spch_proj = [BN_layer1(spch_proji) for spch_proji in spch_proj]
    
    spch_proj = [layer_exp1(spch_proji) for spch_proji in spch_proj]
   
    conv_env_layer = tf.keras.layers.Convolution2D(filters_cnn_env, (kerSize_temporal, 1),
                                                   strides=(stride_temporal, 1), activation="relu")
    spch_proj = [conv_env_layer(spch_proji) for spch_proji in spch_proj]
  
    # layer
    BN_layer2 = tf.keras.layers.BatchNormalization()
    spch_proj = [BN_layer2(spch_proji) for spch_proji in spch_proj]
   
    spch_proj = [layer_permute(spch_proji) for spch_proji in spch_proj]
    
    layer_reshape = tf.keras.layers.Reshape((tf.keras.backend.int_shape(spch_proj[0])[1],
                                             tf.keras.backend.int_shape(spch_proj[0])[2] *
                                             tf.keras.backend.int_shape(spch_proj[0])[3]))
    spch_proj = [layer_reshape(spch_proji) for spch_proji in spch_proj]
   
    lstm_spch = tf.keras.layers.LSTM(units_lstm, return_sequences=True, activation= fun_act)
    
    spch_proj = [lstm_spch(spch_proji) for spch_proji in spch_proj]

    self_attention = GlobalSelfAttention(num_heads=2,key_dim=units_lstm)
    spch_proj = [self_attention(spch_proji) for spch_proji in spch_proj]

    layer_dot = util2.DotLayer()
    spch_proj = [layer_dot([output_eeg,spch_proji]) for spch_proji in spch_proj]

    linear_proj_sim = tf.keras.layers.Dense(1, activation="linear")
    cos_proj = [linear_proj_sim(tf.keras.layers.Flatten()(cos_i)) for cos_i in spch_proj]
    out = tf.keras.activations.softmax((tf.keras.layers.Concatenate()(cos_proj)))

    model = tf.keras.Model(inputs=all_inputs, outputs=[out])

    if compile:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            metrics=["accuracy"],
            loss=["categorical_crossentropy"],
        )
        print(model.summary())
    return model

    return model


def dilation_model(
    time_window=None,
    eeg_input_dimension=64,
    env_input_dimension=1,
    layers=1,
    kernel_size=3,
    spatial_filters=8,
    dilation_filters=16,
    activation="relu",
    compile=True,
    num_mismatched_segments=2
):
    """Convolutional dilation model.

    Code was taken and adapted from
    https://github.com/exporl/eeg-matching-eusipco2020

    Parameters
    ----------
    time_window : int or None
        Segment length. If None, the model will accept every time window input
        length.
    eeg_input_dimension : int
        number of channels of the EEG
    env_input_dimension : int
        dimemsion of the stimulus representation.
        if stimulus == envelope, env_input_dimension =1
        if stimulus == mel, env_input_dimension =28
    layers : int
        Depth of the network/Number of layers
    kernel_size : int
        Size of the kernel for the dilation convolutions
    spatial_filters : int
        Number of parallel filters to use in the spatial layer
    dilation_filters : int
        Number of parallel filters to use in the dilation layers
    activation : str or list or tuple
        Name of the non-linearity to apply after the dilation layers
        or list/tuple of different non-linearities
    compile : bool
        If model should be compiled
    inputs : tuple
        Alternative inputs

    Returns
    -------
    tf.Model
        The dilation model


    References
    ----------
    Accou, B., Jalilpour Monesi, M., Montoya, J., Van hamme, H. & Francart, T.
    Modeling the relationship between acoustic stimulus and EEG with a dilated
    convolutional neural network. In 2020 28th European Signal Processing
    Conference (EUSIPCO), 1175–1179, DOI: 10.23919/Eusipco47968.2020.9287417
    (2021). ISSN: 2076-1465.

    Accou, B., Monesi, M. J., hamme, H. V. & Francart, T.
    Predicting speech intelligibility from EEG in a non-linear classification
    paradigm. J. Neural Eng. 18, 066008, DOI: 10.1088/1741-2552/ac33e9 (2021).
    Publisher: IOP Publishing
    """

    eeg = tf.keras.layers.Input(shape=[time_window, eeg_input_dimension])
    stimuli_input = [tf.keras.layers.Input(shape=[time_window, env_input_dimension]) for _ in range(num_mismatched_segments+1)]

    all_inputs = [eeg]
    all_inputs.extend(stimuli_input)


    stimuli_proj = [x for x in stimuli_input]

    # Activations to apply
    if isinstance(activation, str):
        activations = [activation] * layers
    else:
        activations = activation


    # Spatial convolution
    eeg_proj_1 = tf.keras.layers.Conv1D(spatial_filters, kernel_size=1)(eeg)

    # Construct dilation layers
    for layer_index in range(1):
        # dilation on EEG
        eeg_proj_1 = tf.keras.layers.Conv1D(
            dilation_filters,
            kernel_size=kernel_size,
            dilation_rate=kernel_size ** layer_index,
            strides=1,
            activation=activations[layer_index],
        )(eeg_proj_1)

        # Dilation on envelope data, share weights
        env_proj_layer = tf.keras.layers.Conv1D(
            dilation_filters,
            kernel_size=kernel_size,
            dilation_rate=kernel_size ** layer_index,
            strides=1,
            activation=activations[layer_index],
        )

        stimuli_proj = [env_proj_layer(stimulus_proj) for stimulus_proj in stimuli_proj]
    
    eeg_Encoder = Encoder(num_layers=layers,
                         d_model=dilation_filters,
                         num_heads=2,
                         dff=32,
                         length=eeg_proj_1.shape[1])
    stimuli_Encoder = Encoder(num_layers=layers,
                         d_model=dilation_filters,
                         num_heads=2,
                         dff=32,
                         length=eeg_proj_1.shape[1])
    
    eeg_proj_1 = eeg_Encoder(eeg_proj_1)
    stimuli_proj = [stimuli_Encoder(stimulus_proj) for stimulus_proj in stimuli_proj]


    # Comparison
    cos = [tf.keras.layers.Dot(1, normalize=True)([eeg_proj_1, stimulus_proj]) for stimulus_proj in stimuli_proj]

    linear_proj_sim = tf.keras.layers.Dense(1, activation="linear")

    # Linear projection of similarity matrices
    cos_proj = [linear_proj_sim(tf.keras.layers.Flatten()(cos_i)) for cos_i in cos]


    # Classification
    out = tf.keras.activations.softmax((tf.keras.layers.Concatenate()(cos_proj)))


    model = tf.keras.Model(inputs=all_inputs, outputs=[out])

    if compile:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            metrics=["accuracy"],
            loss=["categorical_crossentropy"],
        )
        print(model.summary())
    return model