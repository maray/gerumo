"""
CNN DET: Bayesian Multi Observer in Deterministic mode
======
Convolutional Neural Network for mono and multi-stereo event reconstruction
"""

import numpy as np
import scipy as sp
import pandas as pd
import scipy.stats as st
from tqdm import tqdm
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
                   padding = "same",
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
                [A list or array with the model's predictions,
                List with the latent output values of the second to last model layer,
                the weights of the last layer.]
        """
        model_telescope = self.models[telescope]
        last_layer_weights = model_telescope.layers[-1].get_weights()
        latent_model = self.latent_models[telescope]
        latent_prediction = latent_model.predict(x_i_telescope,verbose=verbose, **kwargs)
        return [model_telescope.predict(x_i_telescope, verbose=verbose, **kwargs), latent_prediction, last_layer_weights]

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
    
    def evaluate(self, test_assembler_generator, return_inputs=False, return_predictions=False):
        """
        evaluate predict points from a generator, return a point predictions table.
        output:
            return_value: Dict with event, prediction and activated telescopes information
            predictions: array with prediction and standard deviation of input events
                [[[event_1_prediction],[event_1_var]],
                [[event_2_prediction],[event_2_var]],
                ...etc.]
        """
        # Evaluation data
        inputs_values = []
        targets_values = []
        event_ids = []
        true_energy = []
        activated_telescopes = []
        predictions = []
        predicted_variance = []
        predictions_points = []
        variance = []
        covariance = []
        number_of_observation_by_event = []
        total_energy_telescope_type = []
        
        # Save original generator parameters
        original_target_mode = test_assembler_generator.target_mode
        original_event_flag = test_assembler_generator.include_event_id
        original_true_energy_flag = test_assembler_generator.include_true_energy
        
        # Set evaluation parameters to generator
        test_assembler_generator.target_mode = "lineal" #TODO: linear
        test_assembler_generator.include_event_id = True
        test_assembler_generator.include_true_energy = True

        # Iterate for each batch
        for x_batch_j, y_batch_j, meta in tqdm(test_assembler_generator):
            # Model predictions
            if len(self.targets) == 2:
                [predictions_batch_j, batch_variance, number_of_observations, total_energy_telescope_type_batch, activated_telescopes_event, batch_covariance] = self.predict(x_batch_j)
            else:
                [predictions_batch_j, batch_variance, number_of_observations, total_energy_telescope_type_batch, activated_telescopes_event] = self.predict(x_batch_j)
            #predicted_values = self.predict(x_batch_j)
            #predictions_batch_j = predicted_values[:,0].astype(np.float64)
            #batch_variance = predicted_values[:,1].astype(np.float64)
            #number_of_observations = predicted_values[:,2,0].astype(np.float64)
            #total_energy_telescope_type_batch = predicted_values[:,3,0].astype(np.float64)
            #activated_telescopes_event = predicted_values[:,4,0]
            #if len(self.targets) == 2:
                #batch_covariance = predicted_values[:,5,0]
            point_predictions_batch_j = self.point_estimation(predictions_batch_j)

            # Update records
            event_ids.extend(meta["event_id"])
            true_energy.extend(meta["true_energy"])#quitar despues
            activated_telescopes.extend(meta["activated_telescopes"])
            targets_values.extend(y_batch_j)
            predictions_points.extend(point_predictions_batch_j)
            variance.extend(batch_variance)
            number_of_observation_by_event.extend(number_of_observations)
            total_energy_telescope_type.extend(total_energy_telescope_type_batch)
            if len(self.targets) == 2:
                covariance.extend(batch_covariance)
            
            # Return inputs
            if return_inputs:
                inputs_values.extend(x_batch_j)

            # Return predictions
            if return_predictions:
                predictions.extend(predictions_batch_j) # problema de dimensiones. Se suma columna en columna en vez de agregar a las 2 columnas
                predicted_variance.extend(batch_variance)
        
        # ReSet original generator parameters
        test_assembler_generator.target_mode = original_target_mode
        test_assembler_generator.include_event_id = original_event_flag
        test_assembler_generator.include_true_energy = original_true_energy_flag
        
        # Save model results    
        results = pd.DataFrame({
            "event_id":                 event_ids,
            # telescopes_ids: ...,
            "true_mc_energy":           true_energy,
            "activated_telescopes":     activated_telescopes,
            "number_of_observations":   number_of_observation_by_event,
        })
        targets_values = np.array(targets_values)
        predictions_points =  np.array(predictions_points)
        variance = np.array(variance)
        total_energy_telescope_type = np.array(total_energy_telescope_type)
        if len(self.targets) == 2:
            covariance = np.array(covariance)
        for i, target in enumerate(test_assembler_generator.targets):
            results[f"true_{target}"] = targets_values[:,i].flatten()
            for j, telescope in enumerate(activated_telescopes_event[0]):
                results[f"pred_{telescope}_{target}"] = predictions_points[:,j,i].flatten()#select all of one data for j telescope and i target
                results[f"variance_{telescope}_{target}"] = variance[:,j,i].flatten()
        for i, telescope in enumerate(activated_telescopes_event[0]):
            if len(self.targets) == 2:
                results[f"covariance_{telescope}"] = covariance[:,i].flatten()
            results[f"total_energy_{telescope}"] = total_energy_telescope_type[:,i].flatten()

        # Return
        return_value = [results]
        if return_inputs:
            results["inputs_values"] = np.arange(len(inputs_values))
            return_value.append(inputs_values)
        if return_predictions:
            results["predictions"] = np.arange(len(predictions))
            return_value.append(predictions)
            return_value.append(predicted_variance)
            return_value.append(total_energy_telescope_type)
            return_value.append(activated_telescopes_event[0])
            if len(self.targets) == 2:
                return_value.append(covariance)
        # results [, inputs_values] [, predictions]
        return return_value if len(return_value) > 1 else return_value[0]
    
    def model_evaluate(self, telescope, test_unit_generator, return_inputs=False, return_predictions=False):
        """
        """
        # Prepare targets points
        inputs_values = []
        targets_values = []
        event_ids = []
        telescopes_ids = []
        telescopes_types = telescope
        true_energy = []
        predictions = []
        predictions_points = []
        for batch_x, batch_t, batch_meta in tqdm(test_unit_generator):
            # Update records
            event_ids.extend(batch_meta["event_id"])
            telescopes_ids.extend(batch_meta["telescope_id"])
            true_energy.extend(batch_meta["true_energy"])
            targets_values.extend(batch_t)

            # Predictions
            [batch_p, batch_latent_values,weights] = self.model_estimation(batch_x, telescope) 
            batch_p_points = self.point_estimation(batch_p)     # hace samples 2 veces
            predictions_points.extend(batch_p_points)
            
            # Return inputs
            if return_inputs:
                inputs_values.extend(zip(batch_x[0], batch_x[1]))

            # Return predictions
            if return_predictions:
                predictions.extend(batch_p)

        # Save model results
        results = pd.DataFrame({
            "event_id":         event_ids,
            "telescope_id":     telescopes_ids,
            "telescope_type":   [telescope]*len(telescopes_ids),
            "true_mc_energy":   true_energy,
        })
        targets_values = np.array(targets_values)
        predictions_points =  np.array(predictions_points)
        for i, target in enumerate(test_unit_generator.targets):
            results[f"true_{target}"] = targets_values[:,i].flatten()
            results[f"pred_{target}"] = predictions_points[:,i].flatten()

        return_value = [results]
        if return_inputs:
            results["inputs_values"] = np.arange(len(inputs_values))
            return_value.append(inputs_values)
        if return_predictions:
            results["predictions"] = np.arange(len(predictions))
            return_value.append(predictions)
        # Return
        # results [, inputs_values] [, predictions]
        if len(return_value) > 1:
            return return_value 
        else:
            return return_value[0]

    #expected value using the assembling of several telescopes  
    #called with 1 event unique id  
    #returns mean and standard deviation (sigma) of the observations
    def assemble(self, y_i_by_telescope, **kwargs):
        #en este momento funciona para 1 telescopio a la vez. se guardan los datos en listas, una fila para cada tipo de telescopio.
        #La varianza está al cuadrado y se debe procesar después.
        total_intensity = []
        variance = []
        covariance = []
        mean = []   #np array of the mean of each target inside of np array for each telescope ej: [[alt_LST, az_LST],[alt_MST,az_MST]]
        total_observations = []
        telescope_list = []
        intensity = kwargs.get("weights")
        for telescope in y_i_by_telescope:
            telescope_list.append(telescope)
            intensity_by_telescope = intensity[telescope]
            [predictions, latent_values, weights] = y_i_by_telescope[telescope]
            if len(predictions) > 1:
                telescope_variance = []
                if self.assemble_mode == "intensity_weighting":
                    cov_matrix = np.cov(latent_values,rowvar=False, aweights= intensity_by_telescope)
                else:
                    cov_matrix = np.cov(latent_values,rowvar=False)
                for i in range(len(self.targets)):
                    var_temp = np.matmul(weights[0][:,i].flatten(),cov_matrix)
                    telescope_variance.append(np.matmul(var_temp,weights[0][:,i]))
                if len(self.targets) == 2:#covariance for the targets
                    var_temp = np.matmul(weights[0][:,0].flatten(),cov_matrix)
                    telescope_covariance = (np.matmul(var_temp,weights[0][:,1]))#covariance azimut, altitude
                    covariance.append(telescope_covariance)
                variance.append(np.array(telescope_variance))
            else:
                variance.append(np.zeros(len(self.targets)))
                if len(self.targets) == 2:
                    covariance.append(0)
                if intensity_by_telescope == 0: #in the event that there are no measurements for a telescope type
                    mean.append(np.zeros(len(self.targets)))
                    total_intensity.append(0)
                    total_observations.append(0)
                    continue
            mean_event = []
            if self.assemble_mode == "intensity_weighting":
                for i in range(len(self.targets)):
                    mean_event.append(np.dot(intensity_by_telescope, predictions[:,i])/np.sum(intensity_by_telescope))
                mean.append(np.array(mean_event))
            else:
                mean.append(np.mean(predictions,axis=0))#not tested
            
            total_intensity.append(np.sum(intensity_by_telescope))
            total_observations.append(len(intensity_by_telescope))
        #variance = variance / total_intensity**2
        #variance = np.sqrt(variance)
        #mean = mean/total_intensity
        #total_observations = np.tile(total_observations,(len(self.targets),1))
        #total_intensity = np.tile(total_intensity,(len(self.targets),1))
        #telescope_list = np.tile(telescope_list,(len(self.targets),1))
        if len(self.targets) == 2:
            #covariance = np.tile(covariance,(len(self.targets),1))
            return [mean, variance, total_observations, total_intensity, telescope_list, covariance]
        return [mean, variance, total_observations, total_intensity, telescope_list]

        #y_i_all = np.concatenate(list(y_i_by_telescope.values()))
        #if self.assemble_mode == "mean":
        #    yi_assembled = self.mean(y_i_all)
        #elif self.assemble_mode == "intensity_weighting":
        #    yi_assembled = self.intensity_weighting(y_i_all, **kwargs)
        #return yi_assembled

    def exec_model_evaluate(self, model, telescope, test_unit_generator, return_inputs=False, return_predictions=False):
        return_values = self.evaluate(test_unit_generator, return_inputs=return_inputs, return_predictions=return_predictions)
        if isinstance(return_values, list):
            results, *others = return_values
            results["telescope"] = telescope
            return [results] + others
        else:
            return_values["telescope"] = telescope
            return return_values

    def mean(self, y_i):
        return np.mean(y_i, axis=0)
    
    def intensity_weighting(self, y_i, weights=None):
        if weights is None:
            return np.mean(y_i, axis=0)
        else:
            w_i_all = np.concatenate(list(weights.values()))
            if w_i_all.sum() == 0: return np.mean(y_i, axis=0)
            return np.average(y_i, weights=w_i_all, axis=0)