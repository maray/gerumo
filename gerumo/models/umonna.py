"""
Umonna
======

Uncertain Multi Observer Neural Network Assembler
with non-parametric distributions models.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Add,
    Conv2D, MaxPooling2D, Conv3D,
    Conv2DTranspose, Conv3DTranspose,
    UpSampling1D, UpSampling2D, UpSampling3D,
    AveragePooling1D, AveragePooling2D, AveragePooling3D,
    Dense, Flatten, Concatenate, Reshape,
    Activation, BatchNormalization, Dropout
)
from tensorflow.keras.regularizers import l2
from . import CUSTOM_OBJECTS
from .assembler import ModelAssembler
from .layers import HexConvLayer, softmax

def calculate_target_shapes(deconv_blocks=5, first_deconv=(3, 3, 3)):
    target_shape = [k * 3**(deconv_blocks - 2) for k in first_deconv]
    return tuple(target_shape)

def calculate_deconv_parameters(target_shapes=(81, 81, 81), max_deconv=8, max_kernel_size=7):
    deconv_blocks = None
    for deconv_blocks_size in range(2, 1+max_deconv):
        first_deconv = [None] * len(target_shapes)
        candidates = [False] * len(target_shapes)
        for i, target_i in enumerate(target_shapes):
            #print(target_i, deconv_blocks_size)
            kernel_size_i = target_i / (3 ** (deconv_blocks_size - 2))
            #print(kernel_size_i)
            candidates[i] = kernel_size_i.is_integer() and (1 < kernel_size_i <= max_kernel_size)
            first_deconv[i] = int(kernel_size_i) 
        if all(candidates):
            return deconv_blocks_size, first_deconv
    raise ValueError("""target_shapes doesn't have a valid combination. Try a target_shapes from this expresion:
    target_shape_i = kernel_size_i * 3 ** (deconv_blocks - 2) 
    With kernel_size_i > 1 and it can be different for each target, and deconv_blocks >= 2.
    """)

def deconvolution_block(front, targets, deconv_blocks, first_deconv, latent_variables):
    for deconv_i in range(deconv_blocks - 1):
        filters = 4**(deconv_blocks - deconv_i - 1)
        if len(targets) == 1:
            if deconv_i == 0:
                conv_transpose = Conv2DTranspose
                get_new_shape = lambda layer: (layer.get_shape()[2],)
                axis = [1]
                kernel_size = (1, first_deconv[0])
                strides = 1
            else:
                kernel_size = (1, 3)
                strides = (1, 3)
        elif len(targets) == 2:
            if deconv_i == 0:
                conv_transpose = Conv2DTranspose
                get_new_shape = lambda layer: (layer.get_shape()[1], layer.get_shape()[2])
                axis = [1, 2]
                kernel_size = first_deconv
                strides = 1
            else:
                kernel_size = 3
                strides = 3
        elif len(targets) == 3:
            if deconv_i == 0:
                front = Reshape((1, 1, 1, latent_variables//2))(front)
                conv_transpose = Conv3DTranspose
                get_new_shape = lambda layer: (layer.get_shape()[1], layer.get_shape()[2], layer.get_shape()[3])
                axis = [1, 2, 3]
                kernel_size = first_deconv
                strides = 1
            else:
                kernel_size = 3
                strides = 3
        front = conv_transpose(filters=filters, kernel_size=kernel_size,
                        strides=strides, padding="valid", output_padding=0,
                        kernel_initializer='he_uniform')(front)
        front = Activation("relu")(front)
        front = BatchNormalization()(front)
    front = conv_transpose(filters=1, kernel_size=1, strides=1, output_padding=0)(front)
    shape = get_new_shape(front)
    front = Reshape(shape)(front)
    output =  softmax(front, axis=axis)
    return output

def upsampling_block(front, targets, deconv_blocks, first_deconv, latent_variables):
    for deconv_i in range(deconv_blocks - 1):
        filters = 4**(deconv_blocks - deconv_i - 1)
        if len(targets) == 1:
            if deconv_i == 0:
                upsampling = UpSampling2D
                average = AveragePooling2D
                conv = Conv2D
                get_new_shape = lambda layer: (layer.get_shape()[2],)
                axis = [1]
                kernel_size = (1, first_deconv[0])
            else:
                kernel_size = (1, 3)
        elif len(targets) == 2:
            if deconv_i == 0:
                upsampling = UpSampling2D
                average = AveragePooling2D
                conv = Conv2D
                get_new_shape = lambda layer: (layer.get_shape()[1], layer.get_shape()[2])
                axis = [1, 2]
                kernel_size = first_deconv
            else:
                kernel_size = 3
        elif len(targets) == 3:
            if deconv_i == 0:
                front = Reshape((1, 1, 1, latent_variables//2))(front)
                upsampling = UpSampling3D
                average = AveragePooling3D
                conv = Conv3D
                get_new_shape = lambda layer: (layer.get_shape()[1], layer.get_shape()[2], layer.get_shape()[3])
                axis = [1, 2, 3]
                kernel_size = first_deconv
            else:
                kernel_size = 3
        front = upsampling(size=kernel_size, name=f"decoder_upsampling_layer_{deconv_i}")(front)
        front = average(kernel_size, 1, "same", name=f"decoder_average_layer_{deconv_i}")(front)
        front = conv(filters, kernel_size, 1, "same", name=f"decoder_conv_layer_{deconv_i}_a")(front)
        front = Activation("relu", name=f"decoder_ReLU_layer_{deconv_i}_a")(front)
        front = BatchNormalization(name=f"decoder_batchnorm_layer_{deconv_i}_a")(front)
        front = conv(filters, kernel_size, 1, "same", name=f"decoder_conv_layer_{deconv_i}_b")(front)
        front = Activation("relu", name=f"decoder_ReLU_layer_{deconv_i}_b")(front)
        front = BatchNormalization(name=f"decoder_batchnorm_layer_{deconv_i}_b")(front)
    
    deconv_i += 1
    front = conv(filters=1, kernel_size=1, strides=1, name=f"decoder_conv_layer_{deconv_i}")(front)
    shape = get_new_shape(front)
    front = Reshape(shape, name=f"decoder_reshape_layer_{deconv_i}")(front)
    output =  softmax(front, axis=axis)
    return output

def umonna_unit(telescope, image_mode, image_mask, input_img_shape, input_features_shape,
                targets, target_mode, target_shapes=None,
                conv_kernel_sizes=[5, 3, 3], latent_variables=800, dense_layer_blocks=5, deconv_blocks=None, first_deconv=None,
                activity_regularizer_l2=None):
    """Build Umonna Unit Model
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
    # Image Encoding Block
    ## HexConvLayer
    input_img = Input(name="image_input", shape=input_img_shape)
    if image_mode in ("simple-shift", "time-shift"):
        front = HexConvLayer(filters=32, kernel_size=(3,3), name="encoder_hex_conv_layer")(input_img)
    elif image_mode in ("simple", "time"):
        front = Conv2D(name="encoder_conv_layer_0",
                       filters=32, kernel_size=(3,3),
                       kernel_initializer="he_uniform",
                       padding = "valid",
                       activation="relu")(input_img)
        front = MaxPooling2D(name=f"encoder_conv_layer_0", pool_size=(2, 2))(front)
    else:
        raise ValueError(f"Invalid image mode {image_mode}")

    ## convolutional layers
    conv_kernel_sizes = conv_kernel_sizes if conv_kernel_sizes is not None else []
    filters = 32
    conv_i = 0
    for kernel_size in conv_kernel_sizes:
        conv_i += 1
        front = Conv2D(name=f"encoder_conv_layer_{conv_i}_a",
                       filters=filters, kernel_size=kernel_size,
                       kernel_initializer="he_uniform",
                       padding = "same")(front)
        front = Activation(name=f"encoder_ReLU_layer_{conv_i}_a", activation="relu")(front)
        front = BatchNormalization(name=f"encoder_batchnorm_{conv_i}_a")(front)
        
        front = Conv2D(name=f"encoder_conv_layer_{conv_i}_b",
                       filters=filters, kernel_size=kernel_size,
                       kernel_initializer="he_uniform",
                       padding = "same")(front)
        front = Activation(name=f"encoder_ReLU_layer_{conv_i}_b", activation="relu")(front)
        front = BatchNormalization(name=f"encoder_batchnorm_{conv_i}_b")(front)
        front = MaxPooling2D(name=f"encoder_maxpool_layer_{conv_i}", pool_size=(2,2))(front)
        filters *= 2
    
    ## generate latent variables by 1x1 Convolutions
    filters = 2**(5+len(conv_kernel_sizes))
    if telescope == "LST_LSTCam":
        kernel_size = (3, 2)
    elif telescope == "MST_NectarCam":
        kernel_size = (3, 2)
    elif telescope == "MST_FlashCam":
        kernel_size = (5, 1)
    elif telescope == "SST1M_DigiCam":
        kernel_size = (4, 1)

    front = Conv2D(name=f"encoder_conv_layer_{conv_i}",
                   filters=filters, kernel_size=kernel_size,
                   kernel_initializer="he_uniform",
                   padding = "valid")(front)
    front = Activation(name=f"encoder_ReLU_layer_{conv_i}", activation="relu")(front)
    front = BatchNormalization(name=f"encoder_batchnorm_{conv_i}")(front)
    front = Conv2D(name="encoder_conv_layer_to_latent",
                   filters=latent_variables, kernel_size=1,
                   kernel_initializer="he_uniform",
                   padding = "valid",
                   activation="relu")(front)
    front = Flatten(name="encoder_flatten")(front)
    
    l2_ = lambda activity_regularizer_l2: None if activity_regularizer_l2 is None else l2(activity_regularizer_l2)
    # Skip Connection
    # skip_front = front
    # skip_front = Dense(name=f"logic_dense_shortcut", units=latent_variables//2, \
    #                    kernel_regularizer=l2_(activity_regularizer_l2))(skip_front)
    # skip_front = Activation(name=f"logic_ReLU_layer_shortcut", activation="relu")(skip_front)
    # skip_front = BatchNormalization(name=f"logic_batchnorm_shortcut")(skip_front)

    # Logic Block
    ## extra Telescope Features
    input_params = Input(name="feature_input", shape=input_features_shape)
    front = Concatenate()([input_params, front])

    ## dense blocks
    for dense_i in range(dense_layer_blocks):
        front = Dense(name=f"logic_dense_{dense_i}", units=latent_variables//2,  \
                      kernel_regularizer=l2_(activity_regularizer_l2))(front)
        front = Activation(name=f"logic_ReLU_layer_{dense_i}", activation="relu")(front)
        front = BatchNormalization(name=f"logic_batchnorm_{dense_i}")(front)
        front = Dropout(name=f"logic_Dropout_layer_{dense_i}", rate=0.1)(front)

    # Add Skip connection
    #front = Add()([front, skip_front])
    front = Reshape((1, 1,  latent_variables//2), name="logic_reshape")(front)

    # Deconvolution Blocks
    ## calculate deconvolution parameters
    if target_shapes is None:
        if deconv_blocks is None or first_deconv is None:
            raise ValueError("target_shape, deconv_blocks and first_deconv can be None at the same time.")
        target_shapes = calculate_target_shapes(deconv_blocks, first_deconv)
    else:
        #print(target_shapes)
        deconv_blocks, first_deconv = calculate_deconv_parameters(target_shapes)


    ## 
    front = Conv2D(name=f"logic_dense_last", kernel_size=1, 
                   filters=latent_variables//2,
                   kernel_initializer="he_uniform")(front)
    front = Activation(activation="relu")(front)
    front = BatchNormalization()(front)

    if target_mode in ["probability_map", "one_cell", "distance", "lineal"]:
        output = upsampling_block(front, targets, deconv_blocks, first_deconv, latent_variables)
    else:
        raise ValueError(f"Invalid target_mode: '{target_mode}'" )

    model_name = f"Umonna_Unit_{telescope}"
    model = Model(name=model_name, inputs=[input_img, input_params], outputs=output)
    return model

class Umonna(ModelAssembler):
    def __init__(self, sst1m_model_or_path=None, mst_model_or_path=None, mst_nectar_model_or_path = None, lst_model_or_path=None,
                 targets=[], target_domains=tuple(), target_resolutions=tuple(), target_shapes=(),
                 assembler_mode="normalized_product", point_estimation_mode="expected_value", custom_objects=CUSTOM_OBJECTS):
        super().__init__(sst1m_model_or_path=sst1m_model_or_path, mst_model_or_path=mst_model_or_path, mst_nectar_model_or_path = mst_nectar_model_or_path, lst_model_or_path=lst_model_or_path,
                         targets=targets, target_domains=target_domains, target_shapes=target_shapes, custom_objects=custom_objects)
        if assembler_mode not in (None, "normalized_product"):
            raise ValueError(f"Invalid assembler_mode: {assembler_mode}")
        self.assemble_mode = assembler_mode or "normalized_product"
        
        if point_estimation_mode not in ["expected_value", "log_expected_value"]:
            raise ValueError(f"Invalid point_estimation_mode: {point_estimation_mode}")
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
        if self.point_estimation_mode == "expected_value":
            y_point_estimations = self.expected_value(y_predictions)
        elif self.point_estimation_mode == "log_expected_value":
            y_point_estimations = self.log_expected_value(y_predictions)
        return y_point_estimations

    def expected_value(self, y_predictions):
        n_samples = len(y_predictions)
        dimensions = len(self.targets)
        y_point_estimations = np.empty((n_samples, dimensions))
        axis = set(np.arange(dimensions))
        if dimensions == 1:
            indexes_d = np.arange(self.target_shapes[0])
            y_point_estimations[:, 0] = np.array(
                [np.dot(y_i, indexes_d) for y_i in y_predictions]
            )
        else:
            for d in range(dimensions):
                indexes_d = np.arange(self.target_shapes[d])
                reduce_axis = tuple(axis - {d})
                y_point_estimations[:, d] = np.array(
                    [np.dot(y_i.sum(axis=reduce_axis), indexes_d) for y_i in y_predictions]
                )
        # transform from index's domain to target's domain
        target_domains_arr = np.array(self.target_domains)
        target_resolutions_arr = np.array(self.target_resolutions)
        y_point_estimations = (y_point_estimations*target_resolutions_arr) + target_domains_arr[:,0]
        return y_point_estimations

    def log_expected_value(self, y_predictions):
        n_samples = len(y_predictions)
        dimensions = len(self.targets)
        y_point_estimations = np.empty((n_samples, dimensions))
        if dimensions == 1:
            # Transform log domain
            log_y_domain = np.arange(self.target_shapes[0])
            target_domains_arr = np.array(self.target_domains)
            target_resolutions_arr = np.array(self.target_resolutions)
            log_y_domain = (log_y_domain*target_resolutions_arr) + target_domains_arr[:,0]
            y_domain = np.power(10, log_y_domain)
            # Expected value
            y_point_estimations[:, 0] = np.log10(np.array(
                [np.dot(y_domain, prob_i) for prob_i in y_predictions]
            ))
        else:
            raise ValueError("Expected 1-D target. Got", dimensions)
        return y_point_estimations

    def assemble(self, y_i_by_telescope, **kwargs):
        y_i_all = np.concatenate(list(y_i_by_telescope.values()))
        if self.assemble_mode == "normalized_product":
            #yi_assembled = self.normalized_product(y_i_all)
            #elif self.assemble_mode == "normalized_product_with_prior":
            yi_assembled = self.normalized_product_with_prior(y_i_all)
        return yi_assembled
        
    def normalized_product(self, y_i):
        epsilon = 1e-20
        Y_i = np.exp(np.sum(np.log(y_i+epsilon), axis=0))
        Y_i_sum = Y_i.sum()
        if Y_i_sum > 0:
            Y_i /= Y_i.sum()            
        return Y_i

    def normalized_product_with_prior(self, y_i):
        epsilon = 1e-20
        assert y_i.shape[1:] == (81,), "Only support log energy with shape 81"
        y_i_prior = np.log(
            np.array([0.01234376, 0.01234376, 0.01234377, 0.01234379, 0.01234381,
                        0.01234385, 0.01234391, 0.01234398, 0.01234407, 0.01234416,
                        0.01234421, 0.01234436, 0.01234441, 0.01234456, 0.01234469,
                        0.01234482, 0.01234523, 0.01234558, 0.01234595, 0.01234629,
                        0.01234678, 0.01234694, 0.01234709, 0.01234718, 0.01234705,
                        0.0123472 , 0.01234719, 0.0123474 , 0.01234756, 0.01234756,
                        0.01234803, 0.01234822, 0.01234816, 0.01234803, 0.01234791,
                        0.01234814, 0.0123479 , 0.01234776, 0.01234773, 0.0123475 ,
                        0.01234726, 0.01234749, 0.01234749, 0.01234699, 0.01234649,
                        0.01234694, 0.0123466 , 0.01234663, 0.01234642, 0.01234612,
                        0.01234616, 0.01234595, 0.01234601, 0.01234588, 0.01234574,
                        0.01234572, 0.01234544, 0.01234555, 0.01234522, 0.01234556,
                        0.01234531, 0.01234504, 0.01234488, 0.01234482, 0.01234501,
                        0.01234471, 0.01234458, 0.01234469, 0.01234447, 0.01234457,
                        0.01234451, 0.01234465, 0.01234426, 0.01234417, 0.0123441 ,
                        0.01234405, 0.012344  , 0.01234413, 0.01234413, 0.01234396,
                        0.01234399]))
        y_i_ = np.sum(np.log(y_i+epsilon), axis=0) + y_i_prior
        Y_i = np.exp(y_i_)
        Y_i_sum = Y_i.sum()
        if Y_i_sum > 0:
            Y_i /= Y_i.sum()            
        return Y_i
