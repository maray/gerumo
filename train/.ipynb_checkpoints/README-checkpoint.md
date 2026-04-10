# Gerumo Train environment

## Requirements

### Setup Environment with Conda and Linux


Tensorflow with GPU.

```
conda env create -f environment.yml --name gerumo
```

or with CPU only.

```
conda env create -f environment_cpu.yml --name gerumo
```

### Activate 

```
conda activate gerumo
```

## Setup Dataset


* from folder with h5 files
```
(gerumo)$ python prepare_dataset.py --split 0.2 --output  ./dataset --folder ./Prod3b_ML1_gamma
```

* from list of h5 files
```
(gerumo)$ python prepare_dataset.py --split 0.2 --output  ./dataset --files ./Prod3b_ML1_gamma/file_1.h5 ./Prod3b_ML1_gamma/file_2.h5 ...
```

* split: split in train/validation datasets.
* version: prod3b version (default ML1).

## Train a model

Training uses `train dataset` and `validation dataset` defined in configuration files.

```
(gerumo)$ python --config ./config/local/alt_az/umonna_sst.json [--quiet] [-G]
```

* G: Use multigpu training strategy.
* quiet: Training verbosity 0.

### Config files

```
umonna_sst.json
{
    "model_name": "UMONNA_UNIT_MST_V2",  // user defined name
    "model_constructor": "umonna_unit",  // defined in gerumo/models/__init__.py:MODELS
    "assembler_constructor": "umonna",   // defined in gerumo/models/__init__.py:ASSEMBLERS
    "model_extra_params": {
        "latent_variables": 600
    },
    "telescope" : "MST_FlashCam",        // defined in gerumo/data/constants:TELESCOPES
    "output_folder" : "./output",        // user defined output folder
    
    "version": "ML1",                    // ML1 and ML2 prod3b dataset supported
    "train_events_csv"    : "./dataset/train_events.csv",      // train dataset: events 
    "train_telescope_csv" : "./datasets/train_telescopes.csv", // train dataset: telescopes 
    "replace_folder_train" : null,                             // train dataset: folder with h5 files.
    "validation_events_csv"    : "./datasets/validation_events.csv",     // validation dataset: events 
    "validation_telescope_csv" : "./datasets/validation_telescopes.csv", // validation dataset: telescopes 
    "replace_folder_validation" : null,                                  // validation dataset: folder with h5 files.
    "test_events_csv"    : "./datasets/test_events.csv",     // test dataset: events 
    "test_telescope_csv" : "./datasets/test_telescopes.csv", // test dataset: telescopes 
    "replace_folder_test" : null,                            // test dataset: folder with h5 files.
    
    "input_image_mode" : "simple-shift",  // options: ('simple', 'simple-shift', 'time', 'time-shift', 'raw' )
    "input_image_mask" : true,     // include mask channel
    "min_observations" : 1,        // filter dataset by number of observations
    "input_features" : ["x", "y"], // input telescope features
    
    "targets" : ["alt", "az"],     // options: ('alt', 'az', 'log10_mc_energy')+
    "target_mode" : "one_cell",    // options: ('linal', 'one_cell')
    "target_shapes" : {            // umonna option: output shape
        "alt": 81, 
        "az": 81, 
        "log10_mc_energy": 81
    },
    "target_domains" : {           // user defined
        "alt": [1.15, 1.3], 
        "az": [-0.25, 0.25], 
        "log10_mc_energy": [-2.351, 2.47]
    },
    "target_sigmas" : null,        // (deprecated) 
 
    "preprocessing": {             // preprocessing options
        "CameraPipe" : {           // manually tuned MST parameters
            "charge_scaler_value" : 62.652400970459,
            "peak_scaler_path" : "ML1_MST_FlashCam_peak_scaler",
            "tailcuts_clean_params": {"boundary_thresh": 6, "picture_thresh": 14}
        },
        "TelescopeFeaturesPipe": { // ML1 default
            "array_scaler_path": "ML1_array-scaler"
        }
    },

    // Trainging parameters
    "batch_size" : 64,             
    "epochs" : 50,
    "loss" : "crossentropy",
    "optimizer": {
        "learning_rate": 1e-3,
        "name": "sgd",
        "extra_parameters": {
            "momentum": 0.01,
            "nesterov": true
        }
    },
    // callbacks flags
    "save_checkpoints" : true,
    "save_predictions" : true,
    "save_regressions" : true
    
    // (deprecated)
    "summary" : false,
    "plot_only" : false,
    "save_plot" : false,
 }
```
 

## Evaluate

How to evaluate your trained model and assemble results.

Evaluate uses `test dataset` definen in configuration files, if model configuration doesn`t have `test dataset`, uses `validation dataset`.

### Evaluate single model

```
(gerumo)$ python evaluate.py --model  "umonna_model.h5"
                             --config "./config/umonna_model.json"
                             --output "./evaluation"
                             --samples --results --predictions
```


### Evaluate experiment folder
```
(gerumo)$ python evaluate.py --experiment "./umonna_model_run_id"
                             --samples --results --predictions
```
### Evaluate Assembler
```
(gerumo)$ python evaluate.py --assembler "./config/umonna_assembler.json"
                             --output "." 
                             --samples --results --predictions
```
## Extras

### tools.py

* Regression to video.

```
(gerumo)$ python tools.py -i experiment_folder -o output_folder 
```


### debug.py

* Test tensorflow installation with GPU support
```
(gerumo)$ python debug.py --gpu
```

* Plot model
```
(gerumo)$ python debug.py --plot model_checkpoint_or_experiment_folder -o output_folder
```

### prepare_preprocessing.py

TBA
