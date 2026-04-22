"""
Data and dataset constants
==========================
"""
from os import path
from astropy.units.cds import rad, deg, eV
import numpy as np


__all__ = [ 
    'TELESCOPES', 'TELESCOPES_ALIAS',  'TELESCOPES_INVERSE_ALIAS', 'TELESCOPE_FEATURES', 'TELESCOPE_CAMERA',
    'TARGETS', 'TARGET_UNITS',
    'IMAGES_SIZE', 'INPUT_SHAPE', 'PIXELS_POSITION'
]


# Telescopes types
TELESCOPES = ["LST_LSTCam", "MST_NectarCam", "MST_FlashCam", "SST1M_DigiCam"]
TELESCOPES_ALIAS = {
    "ML1": {
        "LST_LSTCam":    "LST", 
        "MST_FlashCam":  "MSTF", 
        "SST1M_DigiCam": "SST1"
    },
    "ML2": {
        "LST_LSTCam":    "LST_LSTCam", 
        "MST_FlashCam":  "MST_FlashCam",
        "SST1M_DigiCam": "SST1M_DigiCam"
    },
    "DL1": {
        "LST_LSTCam":    "LST",
        "MST_NectarCam": "MST",    #Falta este tipo de cámara. Arreglar después
#        "MST_FlashCam":  "MST",
        "SST1M_DigiCam": "SST"
    }
}

TELESCOPES_INVERSE_ALIAS = {
    "LST" : "LST_LSTCam",
    "MST" : "MST_NectarCam"
}

# Telescope array informatin
TELESCOPE_FEATURES = ["x", "y"]
TELESCOPE_CAMERA   = {
        "LST_LSTCam":    "LSTCam", 
        "MST_FlashCam":  "FlashCam", 
        "SST1M_DigiCam": "DigiCam",
        "MST_NectarCam": "NectarCam"
    }

# Regression Targets
TARGETS = ["alt", "az", "mc_energy", "log10_mc_energy"]
TARGET_UNITS = [rad, rad, 1e12*eV, np.log10(1e12)*eV]

# IMAGES Size
IMAGES_SIZE = {
    "LST_LSTCam" : (55, 47),
    "MST_NectarCam" : (55, 47),
    "MST_FlashCam": (84, 29),
    "SST1M_DigiCam": (72, 25)
}

INPUT_SHAPE = {
    "simple": {
        "LST_LSTCam" : (55, 47, 2),
        "MST_NectarCam" : (55, 47, 2),
        "MST_FlashCam": (84, 29, 2),
        "SST1M_DigiCam": (72, 25, 2) 
    },
    "simple-shift": {
        "LST_LSTCam" : (2, 55, 47, 2),
        "MST_NectarCam" : (2, 55, 47, 2),
        "MST_FlashCam": (2, 84, 29, 2),
        "SST1M_DigiCam": (2, 72, 25, 2) 
    },
    "time": {
        "LST_LSTCam" : (55, 47, 1),
        "MST_NectarCam" : (55, 47, 1),
        "MST_FlashCam": (84, 29, 1),
        "SST1M_DigiCam": (72, 25, 1) 
    },
    "time-shift": {
        "LST_LSTCam" : (2, 55, 47, 1),
        "MST_NectarCam" : (2, 55, 47, 1),
        "MST_FlashCam": (2, 84, 29, 1),
        "SST1M_DigiCam": (2, 72, 25, 1) 
    },
    "simple-mask": {
        "LST_LSTCam" : (55, 47, 3),
        "MST_NectarCam" : (55, 47, 3),
        "MST_FlashCam": (84, 29, 3),
        "SST1M_DigiCam": (72, 25, 3)
    },
    "simple-shift-mask": {
        "LST_LSTCam" : (2, 55, 47, 3),
        "MST_NectarCam" : (2, 55, 47, 3),
        "MST_FlashCam": (2, 84, 29, 3),
        "SST1M_DigiCam": (2, 72, 25, 3) 
    },
    "time-mask": {
        "LST_LSTCam" : (55, 47, 2),
        "MST_NectarCam" : (55, 47, 2),
        "MST_FlashCam": (84, 29, 2),
        "SST1M_DigiCam": (72, 25, 2)
    },
    "time-shift-mask": {
        "LST_LSTCam" : (2, 55, 47, 2),
        "MST_NectarCam" : (2, 55, 47, 2),
        "MST_FlashCam": (2, 84, 29, 2),
        "SST1M_DigiCam": (2, 72, 25, 2) 
    }
}

pixpos_folder = path.join(path.dirname(__file__), "pixels_positions")
PIXELS_POSITION = {}
for version in ("ML1", "ML2", "DL1"):
    PIXELS_POSITION[version] = {}
    for mode in ("raw", "simple", "simple_shift", "time", "time_shift"):
        PIXELS_POSITION[version][mode] = {}
        for telescope in TELESCOPES:
            path_  = path.join(pixpos_folder, version, mode, f"{telescope}.npy")
            try:
                pixpos = np.loadtxt(path_, dtype=float)
                pixpos = pixpos.astype(int) if  mode != 'raw' else pixpos
                PIXELS_POSITION[version][mode][telescope] = pixpos
            except OSError as err:
                print(err)
                print("File %s/%s/%s not found" %(version, mode, telescope))