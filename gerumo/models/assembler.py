"""
Multi Observer Assembler
===========
Base models interface for create new multiobservers regresors.
"""

from . import CUSTOM_OBJECTS
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Model
from tqdm import tqdm
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
from os import path
from glob import glob

class ModelAssembler():
    def __init__(self, sst1m_model_or_path=None, mst_model_or_path=None, mst_nectar_model_or_path = None, lst_model_or_path=None,
                    targets=[], target_domains={}, target_shapes=(), custom_objects=CUSTOM_OBJECTS):
        #assert not ((sst1m_model_or_path is None) and (mst_model_or_path is None) and (lst_model_or_path is None)), "No models given" 
        self.models = {}
        self.telescopes = []
        self.custom_objects = custom_objects

        # LOAD SST1M_DigiCam Model
        self.sst_model_path = self.load_model("SST1M_DigiCam", sst1m_model_or_path)

        # LOAD MST_NectarCam Model
        self.mst_nectar_model_path = self.load_model("MST_NectarCam", mst_nectar_model_or_path)

        # LOAD MST_FlashCam Model
        self.mst_model_path = self.load_model("MST_FlashCam", mst_model_or_path)

        # LOAD LST_LSTCam Model
        self.lst_model_path = self.load_model("LST_LSTCam", lst_model_or_path)
       
        self.targets = targets
        self.target_domains = target_domains
        self.target_shapes = target_shapes
        self.mode = ModelAssembler.select_mode(targets)

    @staticmethod
    def select_mode(targets):
        if len(targets) == 1 and targets[0] in ["mc_energy", "log10_mc_energy"]:
            return "energy reconstruction"

        elif len(targets) == 2 and sorted(targets) == sorted(["alt", "az"]):
            return "angular reconstruction"

        elif len(targets) == 3 and\
            (sorted(targets) == sorted(["alt", "az", "mc_energy"]) \
                or sorted(targets) == sorted(["alt", "az", "log10_mc_energy"])):
            return "complete reconstruction"

    def predict(self, x, pbar=None, **kwargs):
        # TODO: duplicate code
        y_predictions = []
        if isinstance(x, list):
            iterator = x if pbar is None else pbar(x)
            for x_i in iterator:
                intensity_i_by_telescope = {}
                y_i_by_telescope = {}
                for telescope in self.telescopes:
                    x_i_telescope = x_i[telescope]
                    # check if events has at least one observation with this telescope type
                    if len(x_i_telescope[0]) > 0:
                        if x_i_telescope[0].ndim == 5:
                            intensity = np.sum(x_i_telescope[0][:, 0, :, :, 0], axis=(1,2))
                        elif x_i_telescope[0].ndim == 4:
                            intensity = np.sum(x_i_telescope[0][:, :, :, 0], axis=(1,2))
                        y_i_by_telescope[telescope] = self.model_estimation(x_i_telescope, telescope, verbose=0, **kwargs)
                        intensity_i_by_telescope[telescope] = intensity
                y_i_assembled = self.assemble(y_i_by_telescope, weights=intensity_i_by_telescope)
                y_predictions.append(y_i_assembled)

        elif isinstance(x, Sequence):
            iterator = x if pbar is None else pbar(x)
            for x_batch_j, _ in iterator:
                for x_i in x_batch_j:
                    intensity_i_by_telescope = {}
                    y_i_by_telescope = {}
                    for telescope in self.telescopes:
                        x_i_telescope = x_i[telescope]
                        # check if events has at least one observation with this telescope type
                        if len(x_i_telescope[0]) > 0:
                            if x_i_telescope[0].ndim == 5:
                                intensity = np.sum(x_i_telescope[0][:, 0, :, :, 0], axis=(1,2))
                            elif x_i_telescope[0].ndim == 4:
                                intensity = np.sum(x_i_telescope[0][:, :, :, 0], axis=(1,2))
                            y_i_by_telescope[telescope] = self.model_estimation(x_i_telescope, telescope, verbose=0, **kwargs)
                            intensity_i_by_telescope[telescope] = intensity
                    y_i_assembled = self.assemble(y_i_by_telescope, weights=intensity_i_by_telescope)
                    y_predictions.append(y_i_assembled)
        return np.array(y_predictions)

    def load_model(self, telescope, model_or_path, custom_objects=None, epoch=None):
        if custom_objects is None:
            custom_objects = self.custom_objects

        if model_or_path is not None:
            self.telescopes.append(telescope)
            if isinstance(model_or_path, str):
                if path.isdir(model_or_path):
                    # Load experiment data
                    experiment_name = path.basename(model_or_path)
                    checkpoints = glob(path.join(model_or_path, "checkpoints", "*.h5"))
                    checkpoints_by_epochs = {int(epoch[-2][1:]) - 1: "_".join(epoch) for epoch in map(lambda s: s.split("_"), checkpoints)}
                    if epoch is None:
                        epoch = max(checkpoints_by_epochs.keys())
                    elif epoch not in checkpoints_by_epochs:
                        epoch = max(filter(lambda e: e < epoch, checkpoints_by_epochs.keys()))
                    # Find epoch model
                    model_name = f"{experiment_name}_e{epoch}"
                    model_or_path = checkpoints_by_epochs[epoch]
                self.models[telescope] = load_model(model_or_path, custom_objects=custom_objects) #keras load model
            elif isinstance(model_or_path, Model):
                self.models[telescope] = model_or_path
        else:
            return self
            

    def predict_point(self, x, **kwargs):
        y_predictions = self.predict(x, **kwargs)
        return self.point_estimation(y_predictions)

    def exec_model_evaluate(self, model, telescope, test_unit_generator, return_inputs=False, return_predictions=False):
        self.models["dummy"] = model
        return_values = self.model_evaluate("dummy", test_unit_generator, return_inputs=return_inputs, return_predictions=return_predictions)
        del self.models["dummy"]
        if isinstance(return_values, list):
            results, *others = return_values
            results["telescope"] = telescope
            return [results] + others
        else:
            return_values["telescope"] = telescope
            return return_values

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
            batch_p = self.model_estimation(batch_x, telescope) 
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
        
    def evaluate(self, test_assembler_generator, return_inputs=False, return_predictions=False):
        """
        evaluate predict points from a generator, return a point predictions table.
        """
        # Evaluation data
        inputs_values = []
        targets_values = []
        event_ids = []
        true_energy = []
        predictions = []
        predictions_points = []
        
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
            predictions_batch_j = self.predict(x_batch_j)
            point_predictions_batch_j = self.point_estimation(predictions_batch_j)

            # Update records
            event_ids.extend(meta["event_id"])
            true_energy.extend(meta["true_energy"])
            targets_values.extend(y_batch_j)
            predictions_points.extend(point_predictions_batch_j)
            
            # Return inputs
            if return_inputs:
                inputs_values.extend(x_batch_j)

            # Return predictions
            if return_predictions:
                predictions.extend(predictions_batch_j)
        
        # ReSet original generator parameters
        test_assembler_generator.target_mode = original_target_mode
        test_assembler_generator.include_event_id = original_event_flag
        test_assembler_generator.include_true_energy = original_true_energy_flag
        
        # Save model results    
        results = pd.DataFrame({
            "event_id":         event_ids,
            # telescopes_ids: ...,
            "true_mc_energy":   true_energy,
        })
        targets_values = np.array(targets_values)
        predictions_points =  np.array(predictions_points)
        for i, target in enumerate(test_assembler_generator.targets):
            results[f"true_{target}"] = targets_values[:,i].flatten()
            results[f"pred_{target}"] = predictions_points[:,i].flatten()

        # Return
        return_value = [results]
        if return_inputs:
            results["inputs_values"] = np.arange(len(inputs_values))
            return_value.append(inputs_values)
        if return_predictions:
            results["predictions"] = np.arange(len(predictions))
            return_value.append(predictions)
        # results [, inputs_values] [, predictions]
        return return_value if len(return_value) > 1 else return_value[0]

    # TODO: change method name
    def exec_model_estimation(self, x_i_telescope, model, verbose=0, **kwargs):
        self.models["dummy"] = model
        prediction = self.model_estimation(x_i_telescope, "dummy",  verbose, **kwargs)
        del self.models["dummy"]
        return prediction
        
    def model_estimation(self, x_i_telescope, telescope, verbose=0, **kwargs):
        raise NotImplementedError

    def point_estimation(self, y_predictions):
        raise NotImplementedError

    def assemble(self, y_i_by_telescope, **kwargs):
        raise NotImplementedError
