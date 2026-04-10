import sys
#FIX: this. re structuring the project
sys.path.insert(1, '..')

from gerumo import *

import argparse
import logging

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Get preprocessing parameters from dataset.")
    ap.add_argument("-i", "--folder", type=str, default="",
                    help="Folder containing hdf5 files.")
    ap.add_argument("-I", "--files", nargs="*", type=str, default=[],
                    help="List of hdf5 files, if is not empty, ignore folder argument.")
    
    args = vars(ap.parse_args())
