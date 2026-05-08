import sys
sys.path.insert(1, '..')
import argparse
from gerumo import *
import numpy as np
from joblib import dump
from tqdm import tqdm
import logging
from sklearn.preprocessing import MinMaxScaler

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate scalers from data")
    ap.add_argument("-e", "--e_file", type=str, default="",
                    help="file containing event data.")
    ap.add_argument("-t", "--t_file", type=str, default="",
                    help="file containing telescope data.")
    ap.add_argument("-v", "--version", type=str, default="DL1",
                    help="Dataset Prod3b version [ML1 or ML2]. DL1 for Prod5.")
    ap.add_argument("-c", "--telescope", type=str, default="array",
                    help="Telescope type with camera. If not included will generate array scaler.")
    ap.add_argument("-a", "--generate_array", action='store_true', dest='array')
    args = vars(ap.parse_args())
    events_path = args["e_file"]
    telescope_path = args["t_file"]
    version = args["version"]
    telescope = args["telescope"]
    array = args["array"]

if events_path is not None and telescope_path is not None:
    load_data = load_dataset(events_path, telescope_path)
    print("Successfully loaded CSV files")
else:
    raise ValueError("Can't find CSV files")

if telescope is "array":
    print("Starting array scaler processing")
    print("Dataset description")
    describe_dataset(load_data)
    telescope_position = load_data[['x', 'y']]
    array_scaler = MinMaxScaler(feature_range=(-1,1)).fit(telescope_position)
    array_scaler.transform(telescope_position.iloc[0].values.reshape((-1,2)))
    dump(array_scaler, f"{version}_array-scaler.gz")
    print("Array scaler successfully created")
else:
    print(f'Starting {telescope} peak scaler creation')
    load_data = aggregate_dataset(load_data, az=True, log10_mc_energy=True)
    load_data = filter_dataset(load_data, version, telescope, 0)
    print("Dataset description")
    describe_dataset(load_data)
    tel_generator = AssemblerUnitGenerator(load_data, 10,
                                        input_image_mode="raw", 
                                        input_image_mask=None, 
                                        input_features=["x","y"],
                                        targets=None,
                                        target_mode=None, 
                                        target_mode_config=None,
                                        version=version
                                        )
    scaler = MinMaxScaler()
    print("Start peak MinMax adjustement")
    for d in tqdm(tel_generator):
        charge, peak = d[0][0][0]
        peak = np.array(peak).flatten().reshape(-1,1)
        scaler.partial_fit(peak)

    dump(scaler, f"{version}_{telescope}_peak_scaler.gz")
    print(f"Scaler file saves as: {version}_{telescope}_peak_scaler.gz")