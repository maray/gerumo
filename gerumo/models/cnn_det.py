"""
CNN DET: Bayesian Multi Observer in Deterministic mode
======
Convolutional Neural Network for mono and multi-stereo event reconstruction
"""

import numpy as np
import scipy as sp
import scipy.stats as st
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Add, Lambda,
    Conv2D, MaxPooling2D, Conv3D,
    Conv2DTranspose, Conv3DTranspose,
    UpSampling1D, UpSampling2D, UpSampling3D,
    AveragePooling1D, AveragePooling2D, AveragePooling3D,
    Dense, Flatten, Concatenate, Reshape,
    Activation, BatchNormalization, Dropout
)
from tensorflow.keras.regularizers import l1, l2
from . import CUSTOM_OBJECTS
from .assembler import ModelAssembler
from .layers import HexConvLayer, softmax


def cnn_det_unit(telescope, image_mode, image_mask, input_img_shape, input_features_shape,
                targets, target_mode, target_shapes=None,
                conv_kernel_sizes=[5, 3, 3], compress_filters=256, compress_kernel_size=3, 
                latent_variables=200, dense_layer_units=[128, 128, 64],
                kernel_regularizer_l2=None, activity_regularizer_l1=None):
    """Build Deterministic CNN Unit Model
    Parameters
    ==========
        telescope
        image_mode
        image_mask
        input_img_shape
        input_features_shape
        output_mode
        output_shape
    Return
    ======
        keras.Model 
    """
    # Soport lineal only
    if target_mode not in ('lineal', 'linear'):
        raise ValueError(f"Invalid target_mode: '{target_mode}'" )

    # Image Encoding Block
    ## HexConvLayer
    input_img = Input(name="image_input", shape=input_img_shape)
    if image_mode == "simple-shift":
        front = HexConvLayer(filters=32, kernel_size=(3,3), name="encoder_hex_conv_layer")(input_img)
    elif image_mode == "simple":
        front = Conv2D(name="encoder_conv_layer_0",
                       filters=32, kernel_size=(3,3),
                       kernel_initializer="he_uniform",
                       padding = "valid",
                       activation="relu")(input_img)
        front = MaxPooling2D(name=f"encoder_max_poolin_layer_0", pool_size=(2, 2))(front)
    else:
        raise ValueError(f"Invalid image mode {image_mode}")

    ## convolutional layers
    conv_kernel_sizes = conv_kernel_sizes if conv_kernel_sizes is not None else []
    filters = 32
    i = 1
    for kernel_size in conv_kernel_sizes:
        front = Conv2D(name=f"encoder_conv_layer_{i}_a",
                       filters=filters, kernel_size=kernel_size,
                       kernel_initializer="he_uniform",
                       padding = "same")(front)
        front = Activation(name=f"encoder_ReLU_{i}_a", activation="relu")(front)
        front = BatchNormalization(name=f"encoder_batchnorm_{i}_a")(front)
        front = Conv2D(name=f"encoder_conv_layer_{i}_b",
                       filters=filters, kernel_size=kernel_size,
                       kernel_initializer="he_uniform",
                       padding = "same")(front)
        front = Activation(name=f"encoder_ReLU_{i}_b", activation="relu")(front)
        front = BatchNormalization(name=f"encoder_batchnorm_{i}_b")(front)
        front = MaxPooling2D(name=f"encoder_maxpool_layer_{i}", pool_size=(2,2))(front)
        filters *= 2
        i += 1

    ## generate latent variables by  Convolutions
    kernel_size = compress_kernel_size
    filters     = compress_filters
    l1_ = lambda activity_regularizer_l1: None if activity_regularizer_l1 is None else l1(activity_regularizer_l1)
    l2_ = lambda kernel_regularizer_l2: None if kernel_regularizer_l2 is None else l2(kernel_regularizer_l2)
    front = Conv2D(name=f"encoder_conv_layer_compress",
                   filters=filters, kernel_size=kernel_size,
                   kernel_initializer="he_uniform",
                   padding = "valid",
                   activation="relu",
                   kernel_regularizer=l2_(kernel_regularizer_l2),
                   activity_regularizer=l1_(activity_regularizer_l1))(front)
    front = Conv2D(name="encoder_conv_layer_to_latent",
                   filters=latent_variables, kernel_size=1,
                   kernel_initializer="he_uniform",
                   padding = "valid",
                   activation="relu",
                   kernel_regularizer=l2_(kernel_regularizer_l2),
                   activity_regularizer=l1_(activity_regularizer_l1))(front)
    front = Flatten(name="encoder_flatten_to_latent")(front)
    
    # Logic Block
    ## extra Telescope Features
    input_params = Input(name="feature_input", shape=input_features_shape)
    front = Concatenate()([input_params, front])

    ## dense blocks
    for dense_i, dense_units in enumerate(dense_layer_units):
        front = Dense(name=f"logic_dense_{dense_i}", units=dense_units, 
                      kernel_regularizer=l2_(kernel_regularizer_l2),
                      activity_regularizer=l1_(activity_regularizer_l1))(front)
        front = Activation(name=f"logic_ReLU_{dense_i}", activation="relu")(front)
        front = BatchNormalization(name=f"logic_batchnorm_{dense_i}")(front)

    # Outpout block
    output = Dense(len(targets), activation="linear")(front)

    model_name = f"CD15_Unit_{telescope}"
    model = Model(name=model_name, inputs=[input_img, input_params], outputs=output)
    return model

#the class BMO_DET inherits the structure of ModelAssembler class, which is defined in assembler.py
class CNN_DET(ModelAssembler):
    def __init__(self, sst1m_model_or_path=None, mst_model_or_path=None, mst_nectar_model_or_path = None, lst_model_or_path=None,
                 targets=[], target_domains=tuple(), target_resolutions=tuple(), target_shapes=(),
                 assembler_mode="intensity_weighting", point_estimation_mode=None, custom_objects=CUSTOM_OBJECTS):

        super().__init__(sst1m_model_or_path=sst1m_model_or_path, mst_model_or_path=mst_model_or_path, mst_nectar_model_or_path = mst_nectar_model_or_path,\
                         lst_model_or_path=lst_model_or_path,
                         targets=targets, target_domains=target_domains, target_shapes=target_shapes, custom_objects=CUSTOM_OBJECTS)
        
        if assembler_mode not in (None, 'mean', 'intensity_weighting'):
            raise ValueError(f"Invalid assembler_mode: {assembler_mode}")

        self.assemble_mode = assembler_mode or "intensity_weighting"
        self.point_estimation_mode = point_estimation_mode
        self.target_resolutions = target_resolutions
    
    def model_estimation(self, x_i_telescope, telescope, verbose=0, **kwargs):
        """
        Predict values for a batch of inputs `x_i_telescope` with the `telescope` model.
            
        Parameters
        ----------
        x_i_telescope : `np.ndarray`
            Batch of inputs with shape [(batch_size, [shift], height, width, channels), (batch_size, telescope_features)]
        telescope : `str`
            Telesope 
        verbose : `int`, optional
            Log extra info. (default=0)
        kwargs : `dict`, optinal
            keras.model.predict() kwargs

        Returns
        -------
            Iterable of size batch_size
                A list or array with the model's predictions.
        """
        model_telescope = self.models[telescope]
        return model_telescope.predict(x_i_telescope, verbose=verbose, **kwargs)

    def point_estimation(self, y_predictions):
        """
        Predict points for a batch of predictions `y_predictions` using `self.point_estimation_mode` method.
            
        Parameters
        ----------
        y_predictions : `np.ndarray` or `list`
            Batch of predictions with len batch_size.

        Returns
        -------
            Iterable of size batch_size
                A list or array with the model's  point predictions.
        """
        return y_predictions

    #expected value using the assembling of several telescopes    
    def assemble(self, y_i_by_telescope, **kwargs):
        y_i_all = np.concatenate(list(y_i_by_telescope.values()))
        if self.assemble_mode == "mean":
            yi_assembled = self.mean(y_i_all)
        elif self.assemble_mode == "intensity_weighting":
            yi_assembled = self.intensity_weighting(y_i_all, **kwargs)
        return yi_assembled

    def mean(self, y_i):
        return np.mean(y_i, axis=0)
    
    def intensity_weighting(self, y_i, weights=None):
        if weights is None:
            return np.mean(y_i, axis=0)
        else:
            w_i_all = np.concatenate(list(weights.values()))
            if w_i_all.sum() == 0: return np.mean(y_i, axis=0)
            return np.average(y_i, weights=w_i_all, axis=0)