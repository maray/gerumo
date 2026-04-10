#FIX: this. re structuring the project
import sys
sys.path.insert(1, '..')

import argparse

from gerumo.baseline.energy import EnergyModel
from gerumo.baseline.reconstructor import Reconstructor

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--events_path", type=str, default="../dataset/events.csv",
                    help="Events folder", required=True)
    ap.add_argument("-t", "--telescopes_path", type=str, default="../dataset/telescopes.csv",
                    help="Telescopes folder", required=True)
    ap.add_argument("-T", "--telescope", type=str, default=None)
    ap.add_argument("-o", "--output", type=str, default=None, required=True)
    ap.add_argument("-n", "--n_events", help="Maximum number of events to use", default=None, type=int)
    ap.add_argument("-c", "--hillas_csv", type=str, default=None)
    ap.add_argument("-r", "--energy_regressor", type=str, default=None)
    ap.add_argument("-f", "--replace_folder", type=str, default=None)
    ap.add_argument("-l", "--loose", action="store_true")
    args = ap.parse_args()

    regressor = None
    if args.energy_regressor is not None:
        regressor = EnergyModel.load(args.energy_regressor)

    reco = Reconstructor(args.events_path, args.telescopes_path, replace_folder=args.replace_folder, telescopes=args.telescope)
    reco.reconstruct_all(
        max_events=args.n_events, min_valid_observations=2, energy_regressor=regressor,
        save_to=args.output, save_hillas=args.hillas_csv, loose=args.loose)
