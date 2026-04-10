"""
Data and dataset input and output
=================================

This module handle data files and generate the datasets used
by the models.
"""

import os
from os import path
from glob import glob
from tqdm import tqdm
import uuid
import csv
import collections
import tables
import logging
import pandas as pd
import numpy as np

from . import TARGETS, TELESCOPES, TELESCOPES_ALIAS

__all__ = [
    'extract_data',
    'generate_dataset', 'load_dataset', 'save_dataset', 'split_dataset',
    'filter_dataset', 'aggregate_dataset', 'describe_dataset', 'load_hillas_dataset',
    'aggregate_hillas_dataset'
]

# Table names and atributes
_events_table = {
    "ML1": "Event_Info",
    "ML2": "Events"
}

_event_attributes = {
    "ML1": {
        "event_id": "event_number",
        "core_x": "core_x",
        "core_y": "core_y",
        "alt": "alt",
        "az": "az",
        "h_first_int": "h_first_int",
        "mc_energy": "mc_energy",
    },
    "ML2": {
        "event_id": "event_id",
        "core_x": "core_x",
        "core_y": "core_y",
        "alt": "alt",
        "az": "az",
        "h_first_int": "h_first_int",
        "mc_energy": "mc_energy",
    }
}

_array_info_table = {
    "ML1": "Array_Info",
    "ML2": "Array_Information"
}

_array_attributes = {
    "ML1": {
        "type": "tel_type",
        "telescope_id": "tel_id",
        "x": "tel_x",
        "y": "tel_y",
        "z": "tel_z",
    },
    "ML2": {
        "type": "type",
        "telescope_id": "id",
        "x": "x",
        "y": "y",
        "z": "z",
    }
}

_telescope_table = {
    "ML1": "Telescope_Info",
    "ML2": "Telescope_Type_Information"
}

_telescopes_info_attributes = {
    "ML1": {
        "num_pixels": "num_pixels",
        "type": "tel_type",
        "pixel_pos": "pixel_pos",
    },
    "ML2": {
        "num_pixels": "num_pixels",
        "type": "type",
        "pixel_pos": "pixel_positions",
    }
}

# CSV events data
_event_fieldnames = [
    'event_unique_id',  # hdf5 event identifier
    'event_id',  # Unique event identifier
    'source',  # hfd5 filename
    'folder',  # Container hdf5 folder
    'core_x',  # Ground x coordinate
    'core_y',  # Ground y coordinate
    'h_first_int',  # Height firts impact
    'alt',  # Altitute
    'az',  # Azimut
    'mc_energy'  # Monte Carlo Energy
]

# CSV Telescope events data
_telescope_fieldnames = [
    'telescope_id',  # Unique telescope identifier
    'event_unique_id',  # hdf5 event identifier
    'type',  # Telescope type
    'x',  # x array coordinate
    'y',  # y array coordinate
    'z',  # z array coordinate
    'observation_indice'  # Observation indice in table
]


def extract_data(hdf5_filepath, version='ML2'):
    """Extract data from one hdf5 file."""

    hdf5_file = tables.open_file(hdf5_filepath, "r")
    source = path.basename(hdf5_filepath)
    folder = path.dirname(hdf5_filepath)

    events_data = []
    telescopes_data = []

    # Array data
    array_data = {}
    # Telescopes Ids
    real_telescopes_id = {}
    ## 'activated_telescopes' are not the real id for each telescopes. These activated_telecopes
    ## are indices related to the event table, but not with  the array information table (telescope info).
    ## In the array info table, all telescope (from different types) are indexed together starting from 1.
    ## But they are in orden, grouped by type (lst, mst and then sst). In the other hand, the Event table
    ## has 3 indices starting from 0, for each telescope type. 
    ## 'real_telescopes_id' translate events indices ('activation_telescope_id') to array ids ('telescope_id').

    for telescope in hdf5_file.root[_array_info_table[version]]:
        telescope_type = telescope[_array_attributes[version]["type"]]
        telescope_type = telescope_type.decode("utf-8") if isinstance(telescope_type, bytes) else telescope_type
        telescope_id = telescope[_array_attributes[version]["telescope_id"]]
        # HERE
        if telescope_type not in array_data:
            array_data[telescope_type] = {}
            real_telescopes_id[telescope_type] = []

        array_data[telescope_type][telescope_id] = {
            "id": telescope_id,
            "x": telescope[_array_attributes[version]["x"]],
            "y": telescope[_array_attributes[version]["y"]],
            "z": telescope[_array_attributes[version]["z"]],
        }
        real_telescopes_id[telescope_type].append(telescope_id)

    # add uuid to avoid duplicated event numbers 
    try:
        for i, event in enumerate(hdf5_file.root[_events_table[version]]):
            # Event data
            event_unique_id = uuid.uuid4().hex[:20]
            event_data = dict(
                event_unique_id=event_unique_id,
                event_id=event[_event_attributes[version]["event_id"]],
                source=source,
                folder=folder,
                core_x=event[_event_attributes[version]["core_x"]],
                core_y=event[_event_attributes[version]["core_y"]],
                h_first_int=event[_event_attributes[version]["h_first_int"]],
                alt=event[_event_attributes[version]["alt"]],
                az=event[_event_attributes[version]["az"]],
                mc_energy=event[_event_attributes[version]["mc_energy"]]
            )
            events_data.append(event_data)

            # Observations data
            ## For each telescope type
            for telescope_type in TELESCOPES:
                telescope_type_alias = TELESCOPES_ALIAS[version][telescope_type]
                telescope_indices = f"{telescope_type_alias}_indices"
                telescopes = event[telescope_indices]
                # number of activated telescopes
                if version == "ML2":
                    telescope_multiplicity = f"{telescope_type_alias}_multiplicity"
                    multiplicity = event[telescope_multiplicity]
                else:
                    multiplicity = np.sum(telescopes != 0)

                if multiplicity == 0:  # No telescope of this type were activated
                    continue

                # Select activated telescopes
                activation_mask = telescopes != 0
                activated_telescopes = np.arange(len(telescopes))[activation_mask]
                observation_indices = telescopes[activation_mask]

                ## For each activated telescope
                for activate_telescope, observation_indice in zip(activated_telescopes, observation_indices):
                    # Telescope Data
                    real_telescope_id = real_telescopes_id[telescope_type_alias][activate_telescope]
                    telescope_data = dict(
                        telescope_id=real_telescope_id,
                        event_unique_id=event_unique_id,
                        type=telescope_type,
                        x=array_data[telescope_type_alias][real_telescope_id]["x"],
                        y=array_data[telescope_type_alias][real_telescope_id]["y"],
                        z=array_data[telescope_type_alias][real_telescope_id]["z"],
                        observation_indice=observation_indice
                    )
                    telescopes_data.append(telescope_data)
    except KeyboardInterrupt:
        logging.info("Extraction stopped.")
    except Exception as err:
        logging.error(err)
        logging.info("Extraction ended by an error.")
    else:
        logging.info("Extraction ended successfully.")
    finally:
        logging.debug(f"Total events: {len(events_data)}")
        logging.debug(f"Total observations: {len(telescopes_data)}")

    return events_data, telescopes_data


def generate_dataset(files_path=None, folder_path=None, output_folder=".", append=False, version="ML2"):
    """Generate events.csv and telescope.csv files. 

    Files generated contains information about the events and their observations
    and are used to reference the compressed hdf5 files with the data.

    Parameters
    ----------
    files_path : `list` of `str`, optional
        List of path to hdf5 files, use these files to generate the dataset.
        If is None, use `folder_path` parameter. (default=None, which assumes 
        that 'folder_path' is not None)
    folder_path : `str`, optional
        Path to folder containing hdf5 files, use these files to generate the 
        dataset. If is None, use `files_path` parameter. (default=None, which assumes 
        that 'files_path' is not None)
    output_folder : `str`
        Path to folder where dataset files will be saved.
    append : `bool`
        Append new events and telescopes to existing files, otherwise create 
        new file. (default=False)

    Returns
    -------
    `tuple` of `str`
        events.csv and telescope.csv path.
    """
    # assert not ((files_path is None)  and (folder_path is None)), "Use one parameters: files_path and folder_path."
    # assert ((files_path is not None)  and (folder_path is not None)), "Use one parameters: files_path and folder_path."

    # hdf5 files
    if files_path is not None:
        files = files_path
        files = [path.abspath(file) for file in files]
    elif folder_path is not None:
        files = glob(path.join(folder_path, "*.h5"))
        files = [path.abspath(file) for file in files]

    # Check if list is not empty
    if len(files) == 0:
        raise FileNotFoundError
    logging.debug(f"{len(files)} files found.")

    # csv files
    mode = "a" if append else "w"
    events_filepath = path.join(output_folder, "events.csv")
    telescope_filepath = path.join(output_folder, "telescopes.csv")
    events_info_csv = open(events_filepath, mode=mode)
    telescope_info_csv = open(telescope_filepath, mode=mode)

    # csv writers
    telescope_writer = csv.DictWriter(telescope_info_csv, delimiter=";",
                                      fieldnames=_telescope_fieldnames, lineterminator="\n")
    events_writer = csv.DictWriter(events_info_csv, delimiter=';',
                                   fieldnames=_event_fieldnames, lineterminator="\n")

    if not append:
        events_writer.writeheader()
        telescope_writer.writeheader()

    total_events = 0
    total_observations = 0
    for file in tqdm(files):
        logging.info(f"Extracting: {file}")
        events_data, telescopes_data = extract_data(file, version)
        total_events += len(events_data)
        total_observations += len(telescopes_data)
        try:
            events_writer.writerows(events_data)
            telescope_writer.writerows(telescopes_data)
        except KeyboardInterrupt:
            logging.info("Extraction stopped.")
            break
    else:
        logging.info("Extraction ended successfully!")
    logging.info(f"Total events: {total_events}")
    logging.info(f"Total observations: {total_observations}")

    # close files
    telescope_info_csv.close()
    events_info_csv.close()

    return events_filepath, telescope_filepath


def split_dataset(dataset, validation_ratio=0.1):
    """Split dataset in train and validation sets using events and a given ratio. 
    
    This split enforce the restriction of don't mix hdf5 files between sets in a 
    imbalance way, but ignore the balance between telescopes type.
    """

    if not (0 < validation_ratio < 1):
        raise ValueError(f"validation_ratio not in (0,1) range: {validation_ratio}")

    # split by events
    total_events = dataset.event_unique_id.nunique()
    val_events_n = int(total_events * validation_ratio)
    train_events_n = total_events - val_events_n

    # enforce source balance
    dataset = dataset.sort_values("source")

    # split by events
    events = dataset.event_unique_id.unique()
    train_events = events[:train_events_n]
    val_events = events[train_events_n:]

    # new datasets
    train_dataset = dataset[dataset.event_unique_id.isin(train_events)]
    val_dataset = dataset[dataset.event_unique_id.isin(val_events)]

    return train_dataset, val_dataset


def load_dataset(events_path, telescopes_path, replace_folder=None):
    """Load events.csv and telescopes.csv files into dataframes.
    
    Parameters
    ----------
    events_path : `str`
        Path to events.csv file.
    telescopes_path : `str`
        Path to telescopes.csv file.
    replace_folder : `str` or `None`
        Path to folder containing hdf5 files. Replace the folder 
        column from csv file. Usefull if the csv files are shared
        between different machines. Default None, means no change
        applied.

    Returns
    -------
    dataset : `pd.DataFrame`
        Dataset of observations for reference telescope images.
    """
    # Load data
    events_data = pd.read_csv(events_path, delimiter=";")
    telescopes_data = pd.read_csv(telescopes_path, delimiter=";")

    # Change dataset folder
    if replace_folder is not None:
        events_data.folder = replace_folder

    # Join tables
    dataset = pd.merge(events_data, telescopes_data, on="event_unique_id", validate="1:m")

    return dataset


def save_dataset(dataset, output_folder, prefix=None):
    """Save events and telescopes dataframes in the corresponding csv files.
        Parameters
    ----------
    dataset : `pd.Dataframe`
        Dataset of observations for reference telescope images.
    output_folder : `str`
        Path to folder where dataset files will be saved.
    prefix : `str`, optional
        Add a prefix to output files names.

    Returns
    -------
    `tuple` of `str`
        events.csv and telescope.csv path.
    """

    event_drop = [field for field in _telescope_fieldnames if field != 'event_unique_id']
    telescope_drop = [field for field in _event_fieldnames if field != 'event_unique_id']

    telescope_data = dataset.drop(columns=telescope_drop)
    event_data = dataset.drop(columns=event_drop)
    event_data = event_data.drop_duplicates()

    event_path = "event.csv" if prefix is None else f"{prefix}_events.csv"
    event_path = path.join(output_folder, event_path)

    telescope_path = "telescopes.csv" if prefix is None else f"{prefix}_telescopes.csv"
    telescope_path = path.join(output_folder, telescope_path)

    event_data.to_csv(event_path, sep=";", index=False)
    telescope_data.to_csv(telescope_path, sep=";", index=False)

    return event_path, telescope_path


"""
Utils Functions
===============
"""


def describe_dataset(dataset, save_to=None):
    files = dataset.source.nunique()
    events = dataset.event_unique_id.nunique()
    obs = len(dataset)
    by_telescope = dataset.type.value_counts()
    print('files', files)
    print('events', events)
    print('observations', obs)
    print('obsevation by telescopes')
    print(by_telescope)

    if save_to is not None:
        with open(save_to, 'w') as save_file:
            save_file.write(f'files: {files}\n')
            save_file.write(f'events: {events}\n')
            save_file.write(f'observations: {obs}\n')
            save_file.write('obsevation by telescopes:\n')
            save_file.write(by_telescope.to_string())


def aggregate_dataset(dataset, az=True, log10_mc_energy=True, hdf5_file=True):
    """
    Perform simple aggegation to targe columns.

    Parameters
    ==========
    az : `bool`, optional
        Translate domain from [0, 2\pi] to [-\pi, \pi]. (default=False)
    log10_mc_energy : `bool`, optional
        Add new log10_mc_energy column, with the logarithm values of mc_energy.
    Returns
    =======
    `pd.DataFrame`
        Dataset with aggregate information.
    """
    if az:
        dataset["az"] = dataset["az"].apply(lambda rad: np.arctan2(np.sin(rad), np.cos(rad)))
    if log10_mc_energy:
        dataset["log10_mc_energy"] = dataset["mc_energy"].apply(lambda energy: np.log10(energy))
    if hdf5_file:
        dataset["hdf5_filepath"] = dataset[["folder", "source"]].apply(lambda x: path.join(x[0], x[1]), axis=1)
    return dataset


def filter_dataset(dataset, telescopes=[], number_of_observations=[], domain={}):
    """
    Select a subset from the dataset given some restrictions.

    The dataset can be filtered by telescope types, number of observations,
    and by a range of values for the targets. 

    Parameters
    ==========
    telescopes : `list` of `str` or 'str'
        Selected telescopes type for the dataset. 'str' if is just one.
    number_of_observations : `list` of `int` or 'int
        For each telescope in 'telescopes' parameter, the minimum amount
        of observations. 'int' if is just one.
    domain : `dict` [ `str`,  `tuple` of `int`]
        A dictionary with names of columns and their value range.
    Returns
    =======
     `pd.DataFrame`
        Filtered dataset.
    """
    if isinstance(telescopes, str):
        telescopes = [telescopes]
    if isinstance(number_of_observations, int):
        number_of_observations = [number_of_observations]

    # filter telescopes
    filtered_dataset = dataset[dataset.type.isin(telescopes)]
    # # filter number of observatoins
    # FIXME: 
    # for telescope, observations in zip(telescopes, number_of_observations):
    #     filtered_events = filtered_dataset[filtered_dataset.type == telescope]\
    #                         .groupby("event_unique_id")\
    #                         .filter(lambda g: len(g) >= observations)\
    #                         .event_unique_id.unique()
    #     filtered_dataset = filtered_dataset[filtered_dataset.event_unique_id.isin(filtered_events)]
    # filter domain
    selection = np.ones((len(filtered_dataset),), dtype=bool)
    for target, (vmin, vmax) in domain.items():
        selection &= ((vmin <= filtered_dataset[target]) & (filtered_dataset[target] <= vmax))
    return filtered_dataset[selection]


def load_array_direction(hdf5_filepath: str, version="ML1"):
    if version != "ML1":
        raise NotImplementedError(f"Dataset version {version} is not yet supported")

    hdf5_file = tables.open_file(hdf5_filepath, "r")
    array_directions = dict()
    for telescope in hdf5_file.root[_array_info_table[version]]:
        telescope_id = telescope[_array_attributes[version]["telescope_id"]]
        telescope_type = telescope[_array_attributes[version]["type"]]
        telescope_type = telescope_type.decode("utf-8") if isinstance(telescope_type, bytes) else telescope_type
        if telescope_type not in array_directions:
            array_directions[telescope_type] = {}
        array_directions[telescope_type][telescope_id] = telescope["run_array_direction"]
    hdf5_file.close()
    return array_directions


def load_hillas_dataset(events_csv: str, telescopes_csv: str, results_csv: str, hillas_csv: str) -> pd.DataFrame:
    """
    Perform simple aggegation to targe columns.

    Parameters
    ==========
    events_csv : `str`
        Events dataset
    telescopes_csv : `str`
        Telescopes dataset
    results_csv: `str`
        The hillas reconstruction .csv generated by `baseline.reconstructor.Reconstructor`
    hillas_csv: `str`
        The hillas parameters .csv saved by `baseline.reconstructor.Reconstructor`
    Returns
    =======
    `pd.DataFrame`
        Dataset with event information, hillas reconstructions results and parameters.
    """

    hillas = pd.read_csv(hillas_csv)
    results = pd.read_csv(results_csv)

    results = results.drop(labels=["core_x", "core_y", "alt", "az", "mc_energy"], axis="columns")

    dataset = load_dataset(events_csv, telescopes_csv)
    dataset = aggregate_dataset(dataset, False, True, False)

    hillas.columns = [f"hillas_{it}" if it in ["x", "y"] else it for it in hillas.columns]

    results_hillas = pd.merge(results, hillas, on=["event_unique_id"], validate="1:m")
    full = pd.merge(dataset, results_hillas, on=["event_unique_id", "type", "observation_indice"], validate="1:1")

    return full


def aggregate_hillas_dataset(dataset: pd.DataFrame):
    dataset.loc[:, "log10_intensity"] = np.log10(dataset["intensity"])

    core_x = dataset["pred_core_x"]
    x = dataset["x"]

    core_y = dataset["pred_core_y"]
    y = dataset["y"]

    impact = np.sqrt((core_x - x) ** 2 + (core_y + y) ** 2)
    dataset.loc[:, "log10_impact"] = np.log10(impact)

    return dataset
