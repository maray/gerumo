import numpy as np
from glob import glob
from os import path
import json
from .data import INPUT_SHAPE
from .data.dataset import load_dataset, aggregate_dataset, filter_dataset
from .data.generator import AssemblerUnitGenerator, AssemblerGenerator
from .data.preprocessing import MultiCameraPipe, CameraPipe, TelescopeFeaturesPipe
from tensorflow.keras.models import load_model
from .models import MODELS, ASSEMBLERS, CUSTOM_OBJECTS


__all__ = [
    'get_target_mode_config',
    'load_dataset_from_experiment', 'load_dataset_from_configuration',
    'load_dataset_from_assembler_configuration',
    'load_model_from_configuration', 'load_model_from_experiment',
    'load_assembler_from_configuration'
]

def __get_resolution(targets, targets_domain, targets_shape):
    """Return the targets resolution for each target given the targets shape"""
    targets_resolution = {}
    for target in targets:
        vmin, vmax = targets_domain[target]
        shape = targets_shape[target]
        targets_resolution[target]  = (vmax -vmin) / shape
    return targets_resolution


def get_target_mode_config(config, target_mode=None):
    targets = config["targets"]
    target_shapes = config["target_shapes"]
    target_domains = config["target_domains"]
    target_mode = target_mode or config["target_mode"]
    if  config["target_shapes"] is not None:
        target_resolutions = __get_resolution(targets, target_domains, target_shapes)
        target_mode_config = {
            "target_shapes":      tuple([target_shapes[target]      for target in targets]),
            "target_domains":     tuple([target_domains[target]     for target in targets]),
            "target_resolutions": tuple([target_resolutions[target] for target in targets])
        }
        if target_mode == "probability_map": #(deprecated)
            target_sigmas = config["target_sigmas"]
            target_mode_config["target_sigmas"] = tuple([target_sigmas[target] for target in targets])
    else:
        target_mode_config = {
            "target_domains":     tuple([target_domains[target]     for target in targets]),
            "target_shapes":      tuple([np.inf      for target in targets]),
            "target_resolutions": tuple([np.inf      for target in targets])
        }
    return target_mode_config


def __same_telescopes(src_telescopes, sample_telescopes):
    return set(sample_telescopes).issubset(set(src_telescopes))


def load_dataset_from_experiment(experiment_folder, include_samples_dataset=False, subset='test'):
    # Find configuration file
    config_file = glob(path.join(experiment_folder, "*.json"))
    if len(config_file) != 1:
        raise ValueError("Config file not found in experiment folder", experiment_folder)
    else:
        config_file = config_file[0]
    return load_dataset_from_configuration(config_file, include_samples_dataset=include_samples_dataset, subset=subset)


def load_dataset_from_configuration(config_file, include_samples_dataset=False, 
                                    subset='test', telescope=None, include_event_id=True, include_true_energy=True,
                                    sample_events=None):
    """"
    Load dataset and generators from experiment configuration.
    
    Parameters
    ==========
    experiment_folder :  `str`or `dict`
        Path to experiment json file or loaded json.
    include_samples_dataset : `bool`
    subset : `bool`
    telescope : `str` or `None`
        Filter dataset for a telescope type, if is `None` it is set by the
        experiments configuration.
    include_event_id : `bool`
        Add event information to 
    include_true_energy : `bool`
        Add true mn_energy column.
    sample_events : `list` or `None`
        List of event ids to for sample dataset. If is `None` it selects random events.
    Returns
    -------
    (`AssemblerUnitGenerator`, `pd.DataFrame`)
        Data generator and dataset, if `include_samples_dataset` is `True` 
        include another tuple with a sample from the same dataset.
    """
    # Load configuration
    if isinstance(config_file, dict):
        config = config_file
    elif isinstance(config_file, str):
        with open(config_file) as cfg_file:
            config = json.load(cfg_file)
    # Prepare datasets
    version = config["version"]

    events_csv    = config[f"{subset}_events_csv"] 
    telescope_csv = config[f"{subset}_telescope_csv"]
    replace_folder_ = config[f"replace_folder_{subset}"] 
    if (events_csv is None) or (telescope_csv is None):
        raise ValueError("Empty datasets, check config file.")
    replace_folder = replace_folder_

    ## Input Parameters 
    telescope = telescope or config["telescope"]
    input_image_mode = config["input_image_mode"]
    input_image_mask = config["input_image_mask"]
    min_observations = config["min_observations"]
    input_features = config["input_features"]
    
    ## Target Parameters 
    targets = config["targets"]
    target_mode = config["target_mode"]
    target_domains = config["target_domains"]
    ## Prepare Generator target_mode_config 
    target_mode_config = get_target_mode_config(config, target_mode=target_mode)

    ## Load Data
    dataset = load_dataset(events_csv, telescope_csv, replace_folder)
    dataset = aggregate_dataset(dataset, az=True, log10_mc_energy=True)

    if include_samples_dataset:
        # events with observations of every type of telescopes
        if sample_events is None:
            sample_telescopes = [telescope]
            sample_events = [e for e, df in dataset.groupby("event_unique_id") if __same_telescopes(df["type"].unique(), sample_telescopes)]
            # TODO: add custom seed
            r = np.random.RandomState(42)
            sample_events = r.choice(sample_events, size=5, replace=False)
        sample_dataset = dataset[dataset["event_unique_id"].isin(sample_events)]
        sample_dataset = filter_dataset(sample_dataset, version, telescope, [0], target_domains)
    else:
        sample_dataset = None
        sample_generator = None
    dataset = filter_dataset(dataset, version, telescope, [0], target_domains)
    
    ## Preprocessing
    preprocessing_parameters = config.get("preprocessing", {})

    # Preprocessing pipes
    ## input preprocessing
    preprocess_input_pipes = {}
    if "CameraPipe" in preprocessing_parameters:
        camera_parameters = preprocessing_parameters["CameraPipe"]
        camera_pipe = CameraPipe(telescope_type=telescope, version=version, **camera_parameters)
        preprocess_input_pipes['CameraPipe'] = camera_pipe
    elif ("MultiCameraPipe" in preprocessing_parameters) and (telescope in preprocessing_parameters["MultiCameraPipe"]):
        camera_parameters = preprocessing_parameters["MultiCameraPipe"][telescope]
        camera_pipe = CameraPipe(telescope_type=telescope, version=version, **camera_parameters)
        preprocess_input_pipes['CameraPipe'] = camera_pipe
        
    if "TelescopeFeaturesPipe" in preprocessing_parameters:
        telescopefeatures_parameters = preprocessing_parameters["TelescopeFeaturesPipe"]
        telescope_features_pipe = TelescopeFeaturesPipe(telescope_type=telescope, version=version, **telescopefeatures_parameters)
        preprocess_input_pipes['TelescopeFeaturesPipe'] = telescope_features_pipe
    ## output preprocessing
    preprocess_output_pipes = {}

    ## Dataset Generators
    generator =  AssemblerUnitGenerator(
                            dataset, 16, 
                            input_image_mode=input_image_mode,
                            input_image_mask=input_image_mask, 
                            input_features=input_features,
                            targets=targets,
                            target_mode=target_mode, 
                            target_mode_config=target_mode_config,
                            preprocess_input_pipes=preprocess_input_pipes,
                            preprocess_output_pipes=preprocess_output_pipes,
                            include_event_id=include_event_id,
                            include_true_energy=include_true_energy,
                            version=version
                        )
    if include_samples_dataset:
        sample_generator =  AssemblerUnitGenerator(
                sample_dataset, min(16, len(sample_dataset)), 
                input_image_mode=input_image_mode,
                input_image_mask=input_image_mask, 
                input_features=input_features,
                targets=targets,
                target_mode=target_mode, 
                target_mode_config=target_mode_config,
                preprocess_input_pipes=preprocess_input_pipes,
                preprocess_output_pipes=preprocess_output_pipes,
                include_event_id=include_event_id,
                include_true_energy=include_true_energy,
                version=version
        )
        return (generator, dataset), (sample_generator, sample_dataset)
    else:
        return generator, dataset
     

def load_dataset_from_assembler_configuration(assembler_config_file, include_samples_dataset=False, subset='test'):
    # Load configuration
    with open(assembler_config_file) as cfg_file:
        config = json.load(cfg_file)
    telescopes = {t:m for t,m in config["telescopes"].items() if m is not None}

    # Prepare datasets
    version = config["version"]
    events_csv    = config[f"{subset}_events_csv"] 
    telescope_csv = config[f"{subset}_telescope_csv"]
    replace_folder_ = config[f"replace_folder_{subset}"]

    ## Input Parameters 
    input_image_mode = config["input_image_mode"]
    input_image_mask = config["input_image_mask"]
    min_observations = config["min_observations"]
    input_features = config["input_features"]
    
    ## Target Parameters 
    targets = config["targets"]
    target_mode = config["target_mode"]
    target_domains = config["target_domains"]
    ## Prepare Generator target_mode_config
    target_mode_config = get_target_mode_config(config, target_mode)

    ## Load Data
    dataset = load_dataset(events_csv, telescope_csv, replace_folder_)
    dataset = aggregate_dataset(dataset, az=True, log10_mc_energy=True)
    if include_samples_dataset:
        # events with observations of every type of telescopes
        sample_telescopes = [t for t in telescopes.keys()]
        sample_events = [e for e, df in dataset.groupby("event_unique_id") if __same_telescopes(df["type"].unique(), sample_telescopes)]
        # TODO: add custom seed
        r = np.random.RandomState(42)
        sample_events = r.choice(sample_events, size=5, replace=False)
        sample_dataset = dataset[dataset["event_unique_id"].isin(sample_events)]
        sample_dataset = filter_dataset(sample_dataset, version, telescopes.keys(), min_observations, target_domains)
        if len(sample_dataset) == 0: raise ValueError("Sample dataset is empty.")
    else:
        sample_events = None
        sample_dataset = None
        sample_generator = None

    dataset = filter_dataset(dataset, version, telescopes.keys(), min_observations, target_domains)
    
    # Evaluate assembler
    ## Preprocessing
    preprocessing_parameters = config.get("preprocessing", {})

    # Preprocessing pipes
    ## input preprocessing
    preprocess_input_pipes = {}
    if ("MultiCameraPipe" in preprocessing_parameters):
        multicamera_parameters = preprocessing_parameters["MultiCameraPipe"]
        multicamera_pipe = MultiCameraPipe(version=version, **multicamera_parameters)
        preprocess_input_pipes['MultiCameraPipe'] = multicamera_pipe
    if "TelescopeFeaturesPipe" in preprocessing_parameters:
        telescopefeatures_parameters = preprocessing_parameters["TelescopeFeaturesPipe"]
        telescope_features_pipe = TelescopeFeaturesPipe(version=version, **telescopefeatures_parameters)
        preprocess_input_pipes['TelescopeFeaturesPipe'] = telescope_features_pipe
    ## output preprocessing
    preprocess_output_pipes = {}

    ## Dataset Generators
    generator =  AssemblerGenerator(
                            dataset, telescopes.keys(), 16, 
                            input_image_mode=input_image_mode,
                            input_image_mask=input_image_mask, 
                            input_features=input_features,
                            targets=targets,
                            target_mode=target_mode, 
                            target_mode_config=target_mode_config,
                            preprocess_input_pipes=preprocess_input_pipes,
                            preprocess_output_pipes=preprocess_output_pipes,
                            include_event_id=True,
                            include_true_energy=True,
                            version=version
                        )
    if include_samples_dataset:
        sample_generator =  AssemblerGenerator(
                sample_dataset, telescopes.keys(), min(16, len(sample_dataset)), 
                input_image_mode=input_image_mode,
                input_image_mask=input_image_mask, 
                input_features=input_features,
                targets=targets,
                target_mode=target_mode, 
                target_mode_config=target_mode_config,
                preprocess_input_pipes=preprocess_input_pipes,
                preprocess_output_pipes=preprocess_output_pipes,
                include_event_id=True,
                include_true_energy=True,
                version=version
        )
        return (generator, dataset), (sample_generator, sample_dataset, sample_events)
    else:
        return generator, dataset


def load_model_from_experiment(experiment_folder, custom_objects=CUSTOM_OBJECTS, 
        assemblers=ASSEMBLERS, models=MODELS, 
        epoch=None, get_assembler=True):
    """"
    Load best model using experiment configuration.
    
    Parameters
    ==========
    experiment_folder :  `str`
        Path to experiment folder.
    epoch : `int`, optional
        Selected epoch, if is in None use the last epoch. (default=None)
    Returns
    -------
    `pd.DataFrame`
        Evaluation results.
    """
    # Load experiment data
    experiment_name = path.basename(experiment_folder)
    checkpoints = glob(path.join(experiment_folder, "checkpoints", "*.h5"))
    checkpoints_by_epochs = {
        int(epoch[-2][1:]) - 1: "_".join(epoch) for epoch in map(lambda s: s.split("_"), checkpoints)
    }
    if epoch is None:
        epoch = max(checkpoints_by_epochs.keys())
    elif epoch not in checkpoints_by_epochs:
        epoch = max(filter(lambda e: e < epoch, checkpoints_by_epochs.keys()))

    # Find epoch model
    model_name = f"{experiment_name}_e{epoch}"
    model_or_path = checkpoints_by_epochs[epoch]
    
    # Find configuration file
    config_file = glob(path.join(experiment_folder, "*.json"))
    if len(config_file) != 1:
        raise ValueError("Config file not found in experiment folder", experiment_folder)
    else:
        config_file = config_file[0]
    return load_model_from_configuration(model_or_path, config_file, 
                                         assemblers=assemblers, custom_objects=custom_objects, 
                                         models=models, model_name=model_name, get_assembler=get_assembler)


def load_model_from_configuration(model_or_path, config_file, 
    custom_objects=CUSTOM_OBJECTS, assemblers=ASSEMBLERS, models=MODELS, 
    model_name=None, telescope=None, get_assembler=True):
    """
    Evaluate model, with configuration file given.
    
    Parameters
    ==========
    model_or_path :  `keras.Model`, `str` or `None`
        Loaded keras model or path to hdf5 checkpoint file. If is `None`, build 
        a new model with `model_constructor` parameter.
    config_file : `str`
        Path to configuration file
    model_name : `str`, optional
        Replace config model name. (default=None)
    Returns
    -------
        config, model [, assembler]
    """
    # Load configuration
    with open(config_file) as cfg_file:
        config = json.load(cfg_file)

    ## Model
    model_name = model_name or config["model_name"]
    telescope = telescope or config["telescope"]
    if model_or_path is None:
        model_constructor = models[config["model_constructor"]]
        input_image_mode = config["input_image_mode"]
        input_image_mask = config["input_image_mask"]
        input_img_shape = INPUT_SHAPE[f"{input_image_mode}-mask" if input_image_mask else input_image_mode][telescope]
        input_features = config["input_features"]
        targets = config["targets"]
        target_mode = config["target_mode"]
        target_mode_config = get_target_mode_config(config, target_mode)
        input_features_shape = (len(input_features),)
        target_shapes = target_mode_config["target_shapes"]
        model_extra_params = config["model_extra_params"]
        model = model_constructor(telescope, input_image_mode, input_image_mask, 
            input_img_shape, input_features_shape,
            config["targets"], config["target_mode"], target_shapes, 
            **model_extra_params)
    elif isinstance(model_or_path, str):
        model = load_model(model_or_path, custom_objects=custom_objects)
    else:
        model = model_or_path 

    if get_assembler:
        targets = config["targets"]
        target_mode = "lineal" #config["target_mode"]
        target_shapes = config["target_shapes"]
        target_domains = config["target_domains"]
        point_estimation_mode = config.get("point_estimation_mode", "expected_value")
        ## Prepare Generator target_mode_config 
        target_mode_config = get_target_mode_config(config, target_mode)
        assembler_constructor = assemblers[config["assembler_constructor"]]
        assembler_mode = config.get("assembler_mode", None)
        assembler = assembler_constructor(
                targets=targets, 
                assembler_mode=assembler_mode,
                target_shapes=target_mode_config["target_shapes"],
                target_domains=target_mode_config["target_domains"],
                target_resolutions=target_mode_config["target_resolutions"],
                point_estimation_mode=point_estimation_mode
        )
        assembler.load_model(telescope, model)
        return config, model, assembler
    else:
        return config, model


def load_assembler_from_configuration(assembler_config_file, assemblers=ASSEMBLERS):
     # Load configuration
    with open(assembler_config_file) as cfg_file:
        config = json.load(cfg_file)
    ## Model
    model_name = config["model_name"]
    model_name = model_name.replace(' ', '_')
    assembler_constructor = assemblers[config["assembler_constructor"]]
    assembler_mode = config.get("assembler_mode", None)
    telescopes = {t:m for t,m in config["telescopes"].items() if m is not None}

    ## Target Parameters 
    targets = config["targets"]
    target_mode = config["target_mode"]
    point_estimation_mode = config.get("point_estimation_mode", "expected_value")
    ## Prepare Generator target_mode_config 
    target_mode_config = get_target_mode_config(config, target_mode)
    # Assembler
    assembler = assembler_constructor(
            targets=targets, 
            assembler_mode=assembler_mode,
            target_shapes=target_mode_config["target_shapes"],
            target_domains=target_mode_config["target_domains"],
            target_resolutions=target_mode_config["target_resolutions"],
            point_estimation_mode=point_estimation_mode
    )
    # Telescope Models 
    for telescope, experiment_or_model_path in telescopes.items():
        assembler.load_model(telescope, experiment_or_model_path)
    return config, assembler
    