"""
BMO: Bayesian Multi Observer
======
Bayesian Neural Network for mono and multi-stereo event reconstruction
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
from ot.lp import free_support_barycenter
from . import CUSTOM_OBJECTS
from .assembler import ModelAssembler
from .layers import HexConvLayer, softmax
from .tools import split_model


def bmo_unit(telescope, image_mode, image_mask, input_img_shape, input_features_shape,
                targets, target_mode, target_shapes=None,
                conv_kernel_sizes=[5, 3, 3], compress_filters=256, compress_kernel_size=3,
                latent_variables=200, dense_layer_units=[128, 128, 64],
                kernel_regularizer_l2=None, activity_regularizer_l1=None,
                dropout_rate=0.5):
    """Build BMO Unit Model
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
    # Support linear target mode only
    if target_mode != 'lineal':
        raise ValueError(f"Invalid target_mode: '{target_mode}'" )

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
        front = MaxPooling2D(name=f"encoder_maxpool_layer_0", pool_size=(2, 2))(front)
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

    ## generate latent variables by Convolutions
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
        if dropout_rate is not None and dropout_rate > 0:
            dense_units = int(dense_units/dropout_rate)
        front = Dense(name=f"logic_dense_{dense_i}", units=dense_units, 
                      kernel_regularizer=l2_(kernel_regularizer_l2),
                      activity_regularizer=l1_(activity_regularizer_l1))(front)
        front = Activation(name=f"logic_ReLU_{dense_i}", activation="relu")(front)
        front = BatchNormalization(name=f"logic_batchnorm_{dense_i}")(front)
        if dropout_rate is not None and dropout_rate > 0:
            front = Dropout(name=f"bayesian_Dropout_{dense_i}", rate=dropout_rate)(front, training=True)

    # Output block
    output = Dense(len(targets), activation="linear", name="regression")(front)
    model_name = f"BMO_Unit_{telescope}"
    model = Model(name=model_name, inputs=[input_img, input_params], outputs=output)
    return model


class BMO(ModelAssembler):
    # static
    cache_models = {}
    def __init__(self, sst1m_model_or_path=None, mst_model_or_path=None, mst_nectar_model_or_path = None, lst_model_or_path=None,
                 targets=[], target_domains=tuple(), target_resolutions=tuple(), target_shapes=(),
                 assembler_mode="resample", point_estimation_mode="expected_value", custom_objects=CUSTOM_OBJECTS):

        super().__init__(sst1m_model_or_path=sst1m_model_or_path, mst_model_or_path=mst_model_or_path, mst_nectar_model_or_path = mst_nectar_model_or_path, lst_model_or_path=lst_model_or_path,
                         targets=targets, target_domains=target_domains, target_shapes=target_shapes, custom_objects=CUSTOM_OBJECTS)
        if assembler_mode not in (
            None, 'resample', 'wasserstein_barycenter', 
            'weighted_mean', 'normalized_product'):
            raise ValueError(f"Invalid assembler_mode: {assembler_mode}")
        if point_estimation_mode not in ["expected_value"]:
            raise ValueError(f"Invalid point_estimation_mode: {point_estimation_mode}")

        self.assemble_mode = assembler_mode or "resample"
        self.point_estimation_mode = point_estimation_mode
        self.target_resolutions = target_resolutions #(deprecated)
        self.sample_size = 500
        self.discrete_bins = {
            "alt": 100,
            "az": 100,
            "log10_mc_energy": 200
        }

    @staticmethod
    def bayesian_estimation_old(model, x_i_telescope, sample_size, verbose, **kwargs):
        """
        Predictive distributions and predicted samples for a batch of inputs
        `x_i_telescope` with model.
            
        Parameters
        ----------
        x_i_telescope : `np.ndarray`
            Batch of inputs with shape
             [(batch_size, [shift], height, width, channels),
             (batch_size, telescope_features)]
        model : `tensorflow.keras.Model`
            Telescope unit model.
        verbose : `int`, optional
            Log extra info. (default=0)
        kwargs : `dict`, optinal
            keras.model.predict() kwargs

        Returns
        -------
            Tuple of list with length batch_size
                Tuple of 2 lists with the model's predictive distributions (kde)
                and samples (np.ndarray).
        """
        y_predictions_points = np.array([model.predict(x_i_telescope, verbose=verbose, **kwargs) for _ in range(sample_size)])
        y_predictions_points = np.swapaxes(np.swapaxes(y_predictions_points, 0, 1), 1, 2)
        y_predictions_kde    = [st.gaussian_kde(y_predictions_point) for y_predictions_point in y_predictions_points]
        return y_predictions_kde, y_predictions_points
    
    @staticmethod
    def bayesian_estimation(model, x_i_telescope, sample_size, verbose, **kwargs):
        """
        Predictive distributions and predicted samples for a batch of inputs
        `x_i_telescope` with model.
            
        Parameters
        ----------
        x_i_telescope : `np.ndarray`
            Batch of inputs with shape
             [(batch_size, [shift], height, width, channels),
             (batch_size, telescope_features)]
        model : `tensorflow.keras.Model`
            Telescope unit model.
        verbose : `int`, optional
            Log extra info. (default=0)
        kwargs : `dict`, optinal
            keras.model.predict() kwargs

        Returns
        -------
            Tuple of list with length batch_size
                Tuple of 2 lists with the model's predictive distributions (kde)
                and samples (np.ndarray).
        """
        # Save time for future predictions
        if model not in BMO.cache_models:
            encoder, regressor = split_model(model, split_layer_name="logic_dense_0")
            BMO.cache_models[model] = {"deterministic": encoder,"stochastic": regressor}
        # Get models
        deterministic, stochastic = BMO.cache_models[model]["deterministic"], BMO.cache_models[model]["stochastic"]
        # Get deterministic latent variables, this part of model is deterministic.
        deterministic_output = deterministic(x_i_telescope)
        # Use the deterministic part for sampling model's stochastic part.
        y_predictions_points = [np.swapaxes(stochastic(np.tile(z_i, (sample_size, 1))), 0, 1) for z_i in deterministic_output]
        # y_predictions_points shape: (input batch_size, dimension, samples batch_size)
        y_predictions_kde    = [st.gaussian_kde(y_predictions_point) for y_predictions_point in y_predictions_points]
        return y_predictions_kde, y_predictions_points

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
        y_predictions, _ = BMO.bayesian_estimation(model_telescope, x_i_telescope, self.sample_size, verbose, **kwargs)
        return y_predictions

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
        return y_point_estimations

    def expected_value(self, y_predictions):
        if isinstance(y_predictions[0], st.gaussian_kde): # resample & wasserstein_barycenter
            y_mus = np.array([y_i.dataset.mean(axis=1) for y_i in y_predictions])
        elif isinstance(y_predictions[0], st._multivariate.multivariate_normal_frozen): # weighted_mean
            y_mus = np.array([y_i.mean for y_i in y_predictions])
        elif isinstance(y_predictions[0], np.ndarray): # normalized_product
            n_samples = len(y_predictions)
            dimensions = len(self.targets)
            y_mus = np.empty((n_samples, dimensions))
            axis = set(np.arange(dimensions))
            for d in range(dimensions):
                reduce_axis = tuple(axis - {d}) if dimensions > 1 else None
                targets_d = np.linspace(
                    self.target_domains[d][0], 
                    self.target_domains[d][1],
                    self.discrete_bins[self.targets[d]]
                )
                y_mus[:, d] = np.array(
                    [np.dot(y_i.sum(axis=reduce_axis), targets_d) for y_i in y_predictions]
                )
        return y_mus

    def assemble(self, y_i_by_telescope, **kwargs):
        y_i_all = np.concatenate(list(y_i_by_telescope.values()))
        if self.assemble_mode == "resample":
            yi_assembled = self.resample(y_i_all)
        elif self.assemble_mode == "normalized_product":
            yi_assembled = self.normalized_product(y_i_all)
        elif self.assemble_mode == "wasserstein_barycenter":
            yi_assembled = self.wasserstein_barycenter(y_i_all)
        elif self.assemble_mode == "weighted_mean":
            yi_assembled = self.weighted_mean(y_i_all)
        return yi_assembled

    def resample(self, y_i):
        #resamples = np.hstack([y_kde_j.resample(self.sample_size) for y_kde_j in y_i])
        resamples = np.hstack([y_kde_j.dataset for y_kde_j in y_i])
        return st.gaussian_kde(resamples)

    def wasserstein_barycenter(self, y_i):
        d = len(self.targets)
        k = self.sample_size                        # number of Diracs of the barycenter
        #X_init = np.random.normal(0., 1., (k, d))  # initial Dirac locations
        X_init = y_i[0].dataset.T
        b = np.ones((k,)) / k
        measures_locations = [y_kde_j.dataset.T for y_kde_j in y_i]
        #measures_weights   = [np.random.uniform(0, 1, (y_kde_j.n,)) for y_kde_j in y_i[:]]
        measures_weights   = [y_kde_j.pdf(y_kde_j.dataset) for y_kde_j in y_i]
        measures_weights   = [b_i/b_i.sum() for b_i in measures_weights]
        # Get Barycenter
        X = free_support_barycenter(measures_locations, measures_weights, X_init, b)        
        return st.gaussian_kde(X.T)

    def normalized_product(self, y_i):
        dimensions = len(self.targets)
        n_samples = len(y_i)
        # space discretization
        if dimensions == 1:
            raise NotImplementedError
        elif dimensions == 2:
            resolution_x = self.discrete_bins[self.targets[0]]
            x_ = np.linspace(
                self.target_domains[0][0], self.target_domains[0][1], resolution_x
            )
            resolution_y = self.discrete_bins[self.targets[1]] 
            y_ = np.linspace(
                self.target_domains[1][0], self.target_domains[1][1], resolution_y
            )
            target_i, target_j = np.meshgrid(x_, y_)
            target_grid = np.dstack((target_i, target_j)).reshape(resolution_x*resolution_y, len(self.targets))
            target_reshape = (resolution_x, resolution_y)
        if dimensions == 3:
            raise NotImplementedError
        # evaluate pdf
        targets_pmf = np.zeros((n_samples, *target_reshape))
        for i in range(n_samples):
            target_pmf = y_i[i].pdf(target_grid.T)
            target_pmf /= target_pmf.sum()
            targets_pmf[i] = target_pmf.reshape(target_reshape).T
        #product
        eps = 1e-16
        Y_i = np.exp(np.sum(np.log(targets_pmf+eps), axis=0))
        Y_i_sum = Y_i.sum()
        Y_i = Y_i/Y_i_sum if Y_i_sum > 0 else Y_i
        Y_i[Y_i < 1e-20] = 0
        return Y_i

    def weighted_mean(self, y_i):
        means = np.array([y_kde_j.dataset.mean(axis=1) for y_kde_j in y_i])
        stds = np.array([y_kde_j.dataset.std(axis=1) for y_kde_j in y_i])
        weighted_mean = np.sum(means * stds, axis=0)/ stds.sum(axis=0)
        dtype = weighted_mean.dtype
        return st.multivariate_normal(
            mean=weighted_mean, 
            cov=np.identity(len(self.targets))*np.finfo(dtype).tiny
        ) # delta dirac distribution
 