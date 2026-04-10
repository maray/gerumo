#FIX: this. re structuring the project
import sys
sys.path.insert(1, '..')

import argparse

import pandas

from gerumo.baseline.energy import EnergyModel
from gerumo.baseline.reconstructor import Reconstructor

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-r", "--results", type=str, help="Results CSV file", required=True)
    ap.add_argument("-p", "--plot_path", type=str, help="Where to save the plots", required=True)
    # ap.add_argument("-o", "--output", type=str, default=None)

    args = ap.parse_args()
    results = pandas.read_csv(args.results)
    Reconstructor.plot(results, args.plot_path)
