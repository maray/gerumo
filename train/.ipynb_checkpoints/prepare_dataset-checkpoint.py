import sys
#FIX: this. re structuring the project
sys.path.insert(1, '..')

from gerumo import *

import argparse
import logging

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Prepare reference dataset for generators.")
    ap.add_argument("-i", "--folder", type=str, default="",
                    help="Folder containing hdf5 files.")
    ap.add_argument("-I", "--files", nargs="*", type=str, default=[],
                    help="List of hdf5 files, if is not empty, ignore folder argument.")
    ap.add_argument("-o", "--output", type=str, default=".", 
                    help="Ouput folder.")
    ap.add_argument("-s", "--split", type=float, default=0.1,
                    help="Validation ratio for split data.")
    ap.add_argument("-v", "--version", type=str, default="ML1",
                    help="Dataset Prod3b version [ML1 or ML2].") 
    ap.add_argument("-a", "--append", dest='append_write', action='store_true')       
    args = vars(ap.parse_args())

    folder = args["folder"]
    files = args["files"]
    output = args["output"]
    split = args["split"]
    append = args["append_write"]
    version = args["version"]

    if len(files) > 0:
        events_path, telescopes_path = generate_dataset(files_path=files, output_folder=output, append=append, version=version)
    elif folder is not None:
        events_path, telescopes_path = generate_dataset(folder_path=folder, output_folder=output, append=append, version=version)
    else:
        raise ValueError("folder or files not set correctly.")

    dataset = load_dataset(events_path, telescopes_path)
    
    print("Dataset")
    describe_dataset(dataset)
    
    if split > 0:
        train_dataset, val_dataset = split_dataset(dataset, split)

        save_dataset(train_dataset, output, "train")
        save_dataset(val_dataset, output, "validation")

        print("\ntrain_dataset:")
        describe_dataset(train_dataset)
        
        print("\nval_dataset:")
        describe_dataset(val_dataset)

    if split == 0:

        save_dataset(dataset, output, "test")
   
        print("\ntest_dataset:")
        describe_dataset(dataset)


