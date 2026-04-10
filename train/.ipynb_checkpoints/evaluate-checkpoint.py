import sys
#FIX: this. re structuring the project
sys.path.insert(1, '..')

from gerumo import *
import json
import logging

import time
import os
from os import path
from os.path import join
from glob import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.metrics import r2_score
from tensorflow.keras.models import load_model, Model
import tensorflow_probability as tfp
import matplotlib as mpl
mpl.use('Agg')


def same_telescopes(src_telescopes, sample_telescopes):
    return set(sample_telescopes).issubset(set(src_telescopes))
    
# Evaluate
# 1. Save results: points, predictions (optional), targets, energy, event info and telescope info
# 2. Calculate regression
# 3. Calculate resolution (angular or energy)
# 4. Prediction Samples (optinal)

def evaluate_experiment_folder(experiment_folder,  save_results=True, save_predictions=True, save_samples=True, epoch=None, replace_folder_test=None, seed=None):
    """"
    Evaluate best model using experiment configuration.
    
    Parameters
    ==========
    experiment_folder :  `str`
        Path to experiment folder.
    save_results : `bool`, optional
        Save predictions points and targets into csv file.
    save_predictions : `bool`, optional
        Save raw predictions.
    save_samples : `bool`, optional
        Save predictions plots from a small events sampleset .
    epoch : `int`, optional
        Selected epoch, if is in None use the last epoch. (default=None)
    replace_folder_test : `str`, optional
        Replace config `replace_folder_test` without edit configuration file.
    seed : `int`, optional
        Seed for random states.
    Returns
    -------
    `pd.DataFrame`
        Evaluation results.
    """
    # Load experiment data
    experiment_name = path.basename(experiment_folder)
    checkpoints = glob(join(experiment_folder, "checkpoints", "*.h5"))
    checkpoints_by_epochs = {int(epoch[-2][1:]) - 1: "_".join(epoch) for epoch in map(lambda s: s.split("_"), checkpoints)}
    if epoch is None:
        epoch = max(checkpoints_by_epochs.keys())
    elif epoch not in checkpoints_by_epochs:
        epoch = max(filter(lambda e: e < epoch, checkpoints_by_epochs.keys()))

    # Find epoch model
    model_name = f"{experiment_name}_e{epoch}"
    model_or_path = checkpoints_by_epochs[epoch]
    
    # Find configuration file
    config_file = glob(join(experiment_folder, "*.json"))
    if len(config_file) != 1:
        raise ValueError("Config file not found in experiment folder", experiment_folder)
    else:
        config_file = config_file[0]
    
    # Evaluate
    output_folder = join(experiment_folder, 'evaluation')
    os.makedirs(output_folder, exist_ok=True)
    print("Making evaluation folder:", output_folder)
    return evaluate_unit(model_or_path, config_file, output_folder,
                  save_predictions=save_predictions, save_samples=save_samples,
                  model_name=model_name, replace_folder_test=replace_folder_test, seed=seed)

def evaluate_unit(model_or_path, config_file, output_folder,
                 assembler=None, telescope=None,
                 save_results=True, save_predictions=True, save_samples=True, 
                 model_name=None, replace_folder_test=None, seed=None, sample_events=None):
    """
    Evaluate model, with configuration file given.
    
    Parameters
    ==========
    model_or_path :  `keras.Model` or `str`
        Loaded keras model or path to hdf5 checkpoint file.
    config_file : `str`
        Path to configuration file
    output_folder : `str`
        Path to folder where plots and results will be saved.
    telescope : `str`, optional
        Model's telescope type.
    assembler : `gerumo.Assembler`, optional
        Assembler model instance.
    save_results : `bool`, optional
        Save predictions points and targets into csv file.
    save_predictions : `bool`, optional
        Save raw predictions.
    save_samples : `bool`, optional
        Save predictions plots from a small events sampleset .
    model_name : `str`, optional
        Replace config model name. (default=None)
    replace_folder_test : `str`, optional
        Replace config `replace_folder_test` without edit configuration file.
    seed : `int`, optional
        Seed for random states.
    sample_events : `list` of `str` or `None`
        Events for sample dataset.
    Returns
    -------
    `pd.DataFrame`
        Evaluation results.
    """
    # Load configuration and model
    config, model, assembler = load_model_from_configuration(
        model_or_path, config_file,
        telescope=telescope,
        assemblers=ASSEMBLERS, 
        custom_objects=CUSTOM_OBJECTS
    )
    # Load Dataset from configuration
    test_, sample_ = load_dataset_from_configuration(
        config_file,
        telescope=telescope,
        include_samples_dataset=True,
        sample_events=sample_events
    )
    (test_generator, test_dataset) = test_
    (sample_generator, sample_dataset) = sample_
    print("Test dataset")
    describe_dataset(test_dataset, save_to=join(output_folder, "test_description.txt"))
    if save_samples:
        print("Sample dataset")
        describe_dataset(sample_dataset, save_to=path.join(output_folder, "sample_description.txt"))
        
    # Load Target configuration
    target_mode_config = get_target_mode_config(config)
    # Model Name
    model_name = model_name or config["model_name"].replace(' ', '_')
    telescope = telescope or config["telescope"]
    # Evaluate
    # 0. Evaluate with generator
    results, predictions = assembler.exec_model_evaluate(model, telescope, test_generator, return_predictions=True)

    # 1. Save results in csv file: points, targets, energy, event info and telescope info
    if save_results:
        results.to_csv(join(output_folder, "results.csv"), index=False)
    if save_predictions:
        # Create predictions folder
        predictions_subfolder = join(output_folder, 'predictions')
        print("Saving predictions in:", predictions_subfolder)
        os.makedirs(predictions_subfolder, exist_ok=True)
        for _, row in results.iterrows():
            prediction_i = row["predictions"]
            prediction = predictions[prediction_i]
            prediction_filepath = join(predictions_subfolder, f"{prediction_i}.npy")
            if isinstance(prediction, np.ndarray):
                np.save(prediction_filepath, prediction)
            elif isinstance(prediction, tfp.distributions.MultivariateNormalTriL) :
                mu = prediction.mean().numpy()
                cov = prediction.covariance().numpy()
                np.save(prediction_filepath, np.vstack((mu, cov)))
            elif isinstance(prediction, st.gaussian_kde):
                np.save(prediction_filepath, prediction.dataset)
            elif isinstance(prediction, st._multivariate.multivariate_normal_frozen):
                np.save(prediction_filepath, prediction.mean)
            else:
                raise NotImplemented("Unknown prediction type:", type(prediction))

    
    # 2. Save samples plots
    if save_samples:
        # makedir
        samples_subfolder = join(output_folder, 'samples')
        print("Saving samples in:", samples_subfolder)
        os.makedirs(samples_subfolder, exist_ok=True)
        _samples = assembler.exec_model_evaluate(
            model, telescope, sample_generator, 
            return_inputs=True, return_predictions=True
        )
        results_samples, inputs_samples, predictions_samples = _samples

        for _, row in results_samples.iterrows():
            # event  and telescope info
            event_id = row["event_id"]
            telescope_id =  row["telescope_id"]
            # input sample
            input_sample = inputs_samples[row["inputs_values"]]
            input_image_sample, input_features_sample = input_sample
            prediction_sample = predictions_samples[row["predictions"]]
            # prediction sample
            # prediction point sample
            pred_targets = [f"pred_{target}" for target in config["targets"]]
            prediction_sample_point = row[pred_targets].to_numpy()
            # Target sample
            true_targets = [f"true_{target}" for target in config["targets"]]
            target_sample = row[true_targets].to_numpy() 
            # Plot input
            image_filepath = join(
                samples_subfolder, 
                f"event_id_{event_id}_telescope_id_{telescope_id}_input.png"
            )
            plot_input_sample(
                input_image_sample, config["input_image_mode"], 
                input_features_sample, title=(event_id, telescope_id),
                make_simple=True, save_to=image_filepath
            )
            # Plot prediction
            prediction_filepath = join(
                samples_subfolder,
                f"event_id_{event_id}_telescope_id_{telescope_id}_prediction.png"
            )
            plot_prediction(
                prediction_sample, prediction_sample_point, config["targets"],
                target_mode_config["target_domains"], 
                target_mode_config["target_resolutions"],
                (event_id, telescope_id), 
                target_sample, save_to=prediction_filepath
            )

    ## 3. Calculate regression
    print("Regression plots")
    scores = r2_score(
        results[[f"true_{target}" for target in config["targets"]]],
        results[[f"pred_{target}" for target in config["targets"]]],
        multioutput="raw_values"
    )
    plot_regression_evaluation(
        results, config["targets"], scores, 
        save_to=join(output_folder, "regression.png")
    )
    reconstruction_mode = assembler.select_mode(config["targets"])
    if reconstruction_mode in ("angular reconstruction", "complete reconstruction"):
        print("Angular Reconstruction")
        ## 3.a Calculate resolution (angular)
        plot_error_and_angular_resolution(
            results, save_to=join(output_folder, "angular_resolution.png")
        )
        plot_angular_resolution_comparison( 
            {telescope: results}, ylim=[0, 2], 
            save_to=join(output_folder,"angular_resolution_comparable.png")
        )

    if reconstruction_mode in ("energy reconstruction", "complete reconstruction"):
        print("Energy resolution")
        ## 3.b Calculate resolution (energy)
        plot_error_and_energy_resolution(results, 
            save_to=join(output_folder, "energy_resolution.png")
        )
        plot_energy_resolution_comparison(
            {telescope: results}, ylim=[0, 2],
            save_to=join(output_folder, "energy_resolution_comparable.png")
        )

    if save_predictions:
        return results, predictions

    return results


def evaluate_assembler(assembler_config_file, output_folder=None, 
                       save_all_unit_evaluations=True, save_results=True, 
                       save_predictions=True, save_samples=True, seed=None):
    """
    Evaluate assembler from a configuration file.
    
    Parameters
    ==========
    assembler_config_file :  `str`
        Path to configuration file.
    output_folder : `str`, optional
        Path to folder where plots and results will be saved 
        (default: use the output from configuration file) 
    save_all_unit_evaluations : `bool`
        Evaluate each telescope.
    save_results : `bool`, optional
        Save predictions points and targets into csv file.
    save_predictions : `bool`, optional
        Save raw predictions.
    save_samples : `bool`, optional
        Save predictions plots from a small events sampleset .
    seed : `int`, optional
        Seed for random states.
    Returns
    -------
    `pd.DataFrame`
        Evaluation results.
    """
    # Load configuration and models
    config, assembler = load_assembler_from_configuration(
        assembler_config_file, assemblers=ASSEMBLERS
    )
    # Load configuration dataset
    test_, sample_ = load_dataset_from_assembler_configuration(
        assembler_config_file, include_samples_dataset=True
    )
    (test_generator, test_dataset) = test_
    (sample_generator, sample_dataset, sample_events) = sample_
    # Load Target configuration
    target_mode_config = get_target_mode_config(config)
    # Model name
    model_name = config["model_name"].replace(' ', '_')
    # Output Folder
    output_folder = output_folder or config["output_folder"]
    output_folder = join(output_folder, f"{model_name}_evaluation")
    print("Saving assembelr evaluation in:", output_folder)
    os.makedirs(output_folder, exist_ok=True)
    with open(join(output_folder, f"{model_name}.json"),  "w") as cfg_file:
        json.dump(config, cfg_file)
    
    print("Test dataset")
    describe_dataset(test_dataset, save_to=join(output_folder, "test_description.txt"))
    if save_samples:
        print("Sample dataset")
        describe_dataset(sample_generator.dataset, save_to=join(output_folder, "sample_description.txt"))

    # Evaluate Model (Optional): Evaluate each telescope model
    if save_all_unit_evaluations:
        all_results = {}
        # Telescope Models 
        for telescope, telescope_model in assembler.models.items():
            model_output_folder = join(output_folder, 'telescopes', telescope)
            print(f"Saving {telescope} evaluation in:", model_output_folder)
            os.makedirs(model_output_folder, exist_ok=True)
            unit_evaluation = evaluate_unit(
                telescope_model, assembler_config_file, model_output_folder,
                assembler=assembler, telescope=telescope, 
                save_results=save_results, save_predictions=save_predictions,
                save_samples=save_samples, model_name=model_name, seed=seed,
                sample_events=sample_events
            )
            if save_predictions:
                all_results[telescope] = unit_evaluation[0]
            else:    
                all_results[telescope] = unit_evaluation
    # Assembler evaluation
    results, predictions = assembler.evaluate(test_generator, return_predictions=True)

    # 1. Save results in csv file: points, targets, energy, event info and telescope info
    if save_results:
        results.to_csv(join(output_folder, "results.csv"), index=False)
    if save_predictions:
        # Create predictions folder
        predictions_subfolder = join(output_folder, 'predictions')
        print("Saving predictions in:", predictions_subfolder)
        os.makedirs(predictions_subfolder, exist_ok=True)
        for _, row in results.iterrows():
            prediction_i = row["predictions"]
            prediction = predictions[prediction_i]
            prediction_filepath = join(predictions_subfolder, f"{prediction_i}.npy")
            if isinstance(prediction, np.ndarray):
                np.save(prediction_filepath, prediction)
            elif isinstance(prediction, tfp.distributions.MultivariateNormalTriL) :
                mu = prediction.mean().numpy()
                cov = prediction.covariance().numpy()
                np.save(prediction_filepath, np.vstack((mu, cov)))
            elif isinstance(prediction, st.gaussian_kde):
                np.save(prediction_filepath, prediction.dataset)
            elif isinstance(prediction, st._multivariate.multivariate_normal_frozen):
                np.save(prediction_filepath, prediction.mean)
            else:
                raise NotImplemented(type(prediction))

    # 2. Save samples plots
    if save_samples:
        # makedir
        samples_subfolder = join(output_folder, 'samples')
        print("Saving samples in:", samples_subfolder)
        os.makedirs(samples_subfolder, exist_ok=True)
        results_samples, predictions_samples = assembler.evaluate(
            sample_generator, return_predictions=True
        )
        for _, row in results_samples.iterrows():
            # event  and telescope info
            event_id = row["event_id"]
            # prediction sample
            prediction_sample = predictions_samples[row["predictions"]]
            # prediction point sample
            pred_targets = [f"pred_{target}" for target in config["targets"]]
            prediction_sample_point = row[pred_targets].to_numpy()
            # Target sample
            true_targets = [f"true_{target}" for target in config["targets"]]
            target_sample = row[true_targets].to_numpy() 

            # Plot prediction
            prediction_filepath = join(
                samples_subfolder, f"event_id_{event_id}_prediction.png"
            )
            plot_prediction(
                prediction_sample, prediction_sample_point, config["targets"],
                target_mode_config["target_domains"], 
                target_mode_config["target_resolutions"], 
                event_id, target_sample, save_to=prediction_filepath
            )

    ## 3. Calculate regression
    print("Regression plots")
    scores = r2_score(
        results[[f"true_{target}" for target in config["targets"]]], 
        results[[f"pred_{target}" for target in config["targets"]]], 
        multioutput="raw_values"
    )
    plot_regression_evaluation(
        results, config["targets"], scores, 
        save_to=join(output_folder, "regression.png")
    )

    reconstruction_mode = assembler.select_mode(config["targets"])
    if reconstruction_mode in ("angular reconstruction", "complete reconstruction"):
        print("Angular Reconstruction")
        ## 3. Calculate resolution (angular)
        plot_error_and_angular_resolution(
            results, save_to=join(output_folder, "angular_resolution.png")
        )
        plot_angular_resolution_comparison(
            {model_name: results}, ylim=[0, 2],
            save_to=join(output_folder, "angular_resolution_comparable.png")
        )
        if save_all_unit_evaluations:
            all_results[model_name] = results
            plot_angular_resolution_comparison(
                all_results, ylim=[0, 2], 
                save_to=join(output_folder, "angular_resolution_comparison.png")
            )

    if reconstruction_mode in ("energy reconstruction", "complete reconstruction"):
        print("Energy resolution")
        ## 3. Calculate resolution (energy)
        plot_error_and_energy_resolution(
            results, save_to=join(output_folder, "energy_resolution.png")
        )
        plot_energy_resolution_comparison(
            {model_name: results},
            ylim=[0, 2], 
            save_to=join(output_folder, "energy_resolution_comparable.png")
        )
        if save_all_unit_evaluations:
            all_results[model_name] = results
            plot_energy_resolution_comparison(
                all_results, ylim=[0, 2], 
                save_to=join(output_folder, "energy_resolution_comparison.png")
            )

    if save_predictions:
        return results, predictions
    return results
    
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Evaluate models.")
    # Option 1
    ap.add_argument("-a", "--assembler", type=str, default=None, 
                    help="Assembler configuration file.")
    ap.add_argument("--all", action="store_true", dest="save_all_unit_evaluations",
                     help="Evaluate each telescope.")
    # Option 2
    ap.add_argument("-e", "--experiment", type=str, default=None, 
                    help="Experiment folder.")
    ap.add_argument("--epoch", type=int, default=None, 
                    help="Select epoch from experiment folder.")
    # Option 3
    ap.add_argument("-m", "--model", type=str, default=None, 
                    help="Model checkpoint. (require configuration file)")
    ap.add_argument("-c", "--config", type=str, default=None, 
                    help="Model configuration file. (require model checkpoint)")
    ## Plots
    ap.add_argument("--results", action="store_true", dest="save_results",
                     help="Save results into csv file.")
    ap.add_argument("--samples", action="store_true", dest="save_samples",
                     help="Save inputs and predictions plots from sample dataset.")
    ap.add_argument("--predictions", action="store_true", dest="save_predictions",
                     help="Save predictions from test dataset.")
    ## Evaluation options
    ap.add_argument("--output", type=str, default=None,
                    help="Output folder.")
    ap.add_argument("--seed", type=int, default=None, 
                    help="Set random state seed. (not implemented)") # TODO: implement seed 
    ap.add_argument("--replace_folder_test", type=str, default=None,
                    help="Replace test folder without edit configuration files.")
    args = vars(ap.parse_args()) 

    # Model/Assembler parameters
    experiment_folder = args["experiment"] 
    assembler_config = args["assembler"]
    model_checkpoint = args["model"]
    model_config = args["config"]

    # Evaluation parameters
    replace_folder_test = args["replace_folder_test"]
    save_predictions = args["save_predictions"]
    save_samples = args["save_samples"]
    save_results = args["save_results"]
    seed = args["seed"]
    
    # Ensemble Evaluation
    if assembler_config is not None:
        output_folder = args["output"]
        evaluate_assembler(assembler_config, 
                          output_folder=output_folder,
                          save_all_unit_evaluations=args["save_all_unit_evaluations"], 
                          save_results=save_results, save_predictions=save_predictions, save_samples=save_samples, 
                          seed=seed)
    # Experiment Evaluation
    elif experiment_folder is not None:
        epoch = args["epoch"]
        evaluate_experiment_folder( experiment_folder, 
                                    save_results=save_results, save_predictions=save_predictions, save_samples=save_samples,
                                    epoch=epoch, replace_folder_test=replace_folder_test, seed=seed)
    # Model Evaluation 
    elif (model_config is not None) and (model_checkpoint is not None):
        output_folder = args["output"] or "."
        output_folder = join(output_folder, 'evaluation')
        os.makedirs(output_folder, exist_ok=True)
        print("Making evaluation folder:", output_folder)
        evaluate_unit( model_checkpoint, model_config, output_folder,
                       save_results=save_results, save_predictions=save_predictions, save_samples=save_samples, 
                       replace_folder_test=replace_folder_test, seed=seed)
    else:
        raise ValueError("Invalid configuration/model input")