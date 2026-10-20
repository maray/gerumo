import sys
#FIX: this. re structuring the project
sys.path.insert(1, '..')

from gerumo import *
import argparse
import os
from os.path import join
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model, Model
import tensorflow_probability as tfp
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('Agg')

os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/lib/cuda'

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate models.")
    ap.add_argument("-a", "--assembler", type=str, default=None, 
                    help="Assembler configuration file.")
    ap.add_argument("-o", "--output", type=str, default=None, 
                        help="output folder")
    
    args = vars(ap.parse_args())
    assembler_config = args["assembler"]
    output_folder = args["output"]
    # Load configuration and models
    config, assembler = load_assembler_from_configuration(
        assembler_config, assemblers=ASSEMBLERS
    )
    # Load configuration dataset
    test_ = load_dataset_from_assembler_configuration(
        assembler_config
    )
    (test_generator, test_dataset) = test_
    


    target_mode_config = get_target_mode_config(config)
    print("Test dataset")
    describe_dataset(test_dataset)
    assembler_evaluation = assembler.evaluate(test_generator, return_predictions=True)
    results = assembler_evaluation[0]
    predictions = assembler_evaluation[1]
    variances = np.asarray(assembler_evaluation[2])
    intensity = assembler_evaluation[3]

    number_of_activated_telescopes = results["number_of_observations"].to_numpy()
    number_of_activated_telescopes = np.array([np.array(x_i) for x_i in number_of_activated_telescopes])

    results_intensity = pd.DataFrame()
    results_variance = pd.DataFrame()
    results_average = pd.DataFrame()
    for t_index, target in enumerate(config["targets"]):
        results_prediction_intensity = None
        results_prediction_variance = None
        results_prediction_average = None
        total_intensity = None
        added_inverted_variance = None
        total_activated_telescopes = None
        for tel_index, telescope in enumerate(assembler.telescopes):
            weighted_prediction_intensity = (results[f"pred_{telescope}_{target}"] * intensity[:, tel_index])
            weighted_prediction_variance = (results[f"pred_{telescope}_{target}"] * number_of_activated_telescopes[:,tel_index] / variances[:,tel_index,t_index])
            average_prediction = (results[f"pred_{telescope}_{target}"] * number_of_activated_telescopes[:,tel_index])
            if results_prediction_intensity is None:
                results_prediction_intensity = weighted_prediction_intensity
                results_prediction_variance = weighted_prediction_variance
                results_prediction_average = average_prediction
                total_intensity = intensity[:,tel_index].copy()
                added_inverted_variance = number_of_activated_telescopes[:,tel_index] / variances[:,tel_index,t_index]
                total_activated_telescopes = number_of_activated_telescopes[:,tel_index].copy()
            else:
                results_prediction_intensity += weighted_prediction_intensity
                results_prediction_variance += weighted_prediction_variance
                results_prediction_average += average_prediction
                total_intensity += intensity[:,tel_index]
                added_inverted_variance += number_of_activated_telescopes[:,tel_index] / variances[:,tel_index,t_index]
                total_activated_telescopes += number_of_activated_telescopes[:,tel_index]
        results_intensity[f"prediction_{target}"] = results_prediction_intensity / total_intensity
        results_variance[f"prediction_{target}"] = results_prediction_variance / added_inverted_variance
        results_average[f"prediction_{target}"] = results_prediction_average / total_activated_telescopes

    error_intensity = []
    error_variance = []
    error_average = []
    
    for i, target in enumerate(config["targets"]):
        error_intensity.append(np.abs(results_intensity[f"prediction_{target}"] - results[f"true_{target}"]))
        error_variance.append(np.abs(results_variance[f"prediction_{target}"] - results[f"true_{target}"]))
        error_average.append(np.abs(results_average[f"prediction_{target}"] - results[f"true_{target}"]))

    average_error_intensity = []
    average_error_variance = []
    average_error_average = []

    for i, target in enumerate(config["targets"]):

        average_error_intensity.append(sum(error_intensity[i])/len(error_intensity[i]))
        average_error_variance.append(sum(error_variance[i])/len(error_variance[i]))
        average_error_average.append(sum(error_average[i])/len(error_average[i]))


    n_targets = len(config["targets"])
    # Create Figure and axis
    name = config["model_name"]
    fig, axs = plt.subplots(1, n_targets, figsize=(n_targets*8.5 ,8))
    plt.suptitle(f"Error for {name} whith Variance and Intensity Ponderation")

    for i, target in enumerate(config["targets"]):
        if n_targets > 1:
            sub = axs[i]
        else:
            sub = axs

        x = error_intensity[i]
        y = error_variance[i]

        max_value_x = np.max(x)
        max_value_y = np.max(y)
        max_value = np.max([max_value_x, max_value_y])

        proportion = y > x
        proportion = proportion.sum()
        proportion = proportion / len(x)

        sub.scatter(x = x, y = y)
        sub.set_title("Error for {0} with Proportion of {1:%} Higher Variance than Intensity Error".format(target,proportion))
        sub.set_xlabel("Intensity ponderation error")
        sub.set_ylabel("Variance ponderation error")
        sub.plot([0, max_value], [0, max_value], "--")

    plt.savefig(join(output_folder, f"error comparisson {name}.png"))
    plt.close()

    width = 0.2
    x = np.arange(n_targets)
    fig, axs = plt.subplots(layout="constrained")
    axs.set_title(f"Average Error for {name}")
    axs.bar(x - width, average_error_average, width)
    axs.bar(x, average_error_intensity, width)
    axs.bar(x + width, average_error_variance, width)

    axs.set_ylabel("Average error.")
    axs.set_xticks(x,config["targets"])
    axs.set_xlabel("Targets")
    axs.legend(["Error average", "Error intensity", "Error variance"])
    plt.savefig(join(output_folder, f"average error comparisson {name}.png"))
    plt.close()

    fig, axs = plt.subplots(n_targets, 3, figsize=(24, n_targets*8.5))
    plt.suptitle(f"Predicted values coparison for each assembly method for {name}")

    for i, target in enumerate(config["targets"]):
        if n_targets > 1:
            target_sub = axs[i]
        else:
            target_sub = axs

        true_targets = results[f"true_{target}"]
        min_val = np.min(true_targets)
        max_val = np.max(true_targets)

        sub = target_sub[0]
        sub.scatter(true_targets, results_average[f"prediction_{target}"])
        sub.plot([min_val, max_val], [min_val, max_val], "--")
        sub.set_xlabel(f"True {target} values")
        sub.set_ylabel(f"Predicted average assembly {target} values")
        sub.set_title("Predicted Values Averaged v/s True Targets")

        sub = target_sub[1]
        sub.scatter(true_targets, results_intensity[f"prediction_{target}"])
        sub.plot([min_val, max_val], [min_val, max_val], "--")
        sub.set_xlabel(f"True {target} values")
        sub.set_ylabel(f"Predicted intensity assembly {target} values")
        sub.set_title("Predicted Values Weighted by Intensity v/s True Targets")

        sub = target_sub[2]
        sub.scatter(true_targets, results_variance[f"prediction_{target}"])
        sub.plot([min_val, max_val], [min_val, max_val], "--")
        sub.set_xlabel(f"True {target} values")
        sub.set_ylabel(f"Predicted intensity assembly {target} values")
        sub.set_title("Predicted Values Weighted by Variance v/s True Targets")

    plt.savefig(join(output_folder, f"predicted values comparisson {name}.png"))
    plt.close()

    intensity_by_telescope = []

    for tel_index, telescope in enumerate(assembler.telescopes):
        intensity_list_by_telescope = intensity[:, tel_index] / number_of_activated_telescopes[:,tel_index]
        intensity_by_telescope.append(sum(intensity_list_by_telescope) / len(intensity_list_by_telescope))

    plt.bar(assembler.telescopes, intensity_by_telescope)
    plt.xlabel("Telescopes")
    plt.ylabel("Intensity")
    plt.title("Average Image Intensity by Telescope")
    plt.savefig(join(output_folder, f"image intensity by telescope.png"))