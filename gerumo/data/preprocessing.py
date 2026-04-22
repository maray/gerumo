from ctapipe.image import tailcuts_clean, dilate
from joblib import load
from . import load_camera_geometry
from os.path import join, exists, dirname
import numpy as np

__all__ = ["CameraPipe", "MultiCameraPipe", "TelescopeFeaturesPipe"]

__scalers_folder = join(dirname(__file__), "scalers")
__default_scalers = {
    "ML1_array-scaler": join(__scalers_folder, "ML1_array-scaler.gz"),
    "ML1_LST_LSTCam_peak_scaler" : join(__scalers_folder, "ML1_LST_LSTCam_peak_scaler.gz"),
    "ML1_MST_FlashCam_peak_scaler" : join(__scalers_folder, "ML1_MST_FlashCam_peak_scaler.gz"),
    "ML1_SST1M_DigiCam_peak_scaler" : join(__scalers_folder, "ML1_SST1M_DigiCam_peak_scaler.gz"),
    "DL1_array-scaler": join(__scalers_folder, "DL1_array-scaler.gz"),
    "DL1_LST_LSTCam_peak_scaler" : join(__scalers_folder, "DL1_LST_LSTCam_peak_scaler.gz"),
    "DL1_MST_NectarCam_peak_scaler" : join(__scalers_folder, "DL1_MST_NectarCam_peak_scaler.gz")
}

def load_scaler(default_name_or_custom_scaler_path):
    if default_name_or_custom_scaler_path in __default_scalers:
        return load(__default_scalers[default_name_or_custom_scaler_path])
    elif default_name_or_custom_scaler_path is None:
        return None
    elif exists(default_name_or_custom_scaler_path):
        return load(default_name_or_custom_scaler_path)
    else:
        raise OSError(f"Scaler not found: {default_name_or_custom_scaler_path}")

class CameraPipe():
    def __init__(self, telescope_type, charge_scaler_value=None, peak_scaler_path=None, tailcuts_clean_params=None, version="ML1"):
        self.telescope_type = telescope_type
        self.charge_scaler_value = charge_scaler_value if charge_scaler_value is not None else 1
        self.peak_scaler = load_scaler(peak_scaler_path)
        self.tailcuts_clean_params = tailcuts_clean_params #{picture_thresh:int, boundary_thresh:int}
        self.geometry = load_camera_geometry(telescope_type, version)
        self.version = version

    def __call__(self, cameras):
        if not isinstance(cameras, list):
            cameras = [cameras]
        results = []
        for (charge, peak) in cameras:
            if self.tailcuts_clean_params is not None:
                cleanmask = tailcuts_clean(self.geometry, charge, **self.tailcuts_clean_params)
                charge[~cleanmask] = 0.0
                for _ in range(3):
                    cleanmask = dilate(self.geometry, cleanmask)
                peak[~cleanmask] = 0.0
            charge = np.divide(charge,self.charge_scaler_value)
            if self.peak_scaler is not None:
                peak  = self.peak_scaler.transform(peak.reshape((-1, 1))).flatten()
            results.append((charge, peak))
        return results


class MultiCameraPipe():
    def __init__(self, SST1M_DigiCam=None, MST_FlashCam=None, LST_LSTCam=None, version="ML1"):
        assert not ((SST1M_DigiCam is None) and (MST_FlashCam is None) and (LST_LSTCam is None)), "No Pipes given" 
        # Camera Pipes and telescope type supported
        self.camera_pipes = {}
        self.telescopes = []

        # Load SST1M CameraPipe
        self.load_camera_pipe("SST1M_DigiCam", SST1M_DigiCam, version)
        # Load MST CameraPipe
        self.load_camera_pipe("MST_FlashCam", MST_FlashCam, version)
        # Load LST CameraPipe
        self.load_camera_pipe("LST_LSTCam", LST_LSTCam, version)
        
        # Dataset Version        
        self.version = version

    def load_camera_pipe(self, telescope_type, camerapipe_or_parameters, version="ML1"):    
        if isinstance(camerapipe_or_parameters, CameraPipe):
            camera_pipe = camerapipe_or_parameters
            if telescope_type != camera_pipe.telescope_type:
                raise ValueError(f"Invalid CameraPipe for {telescope_type}, telescope_type differs CameraPipe.telescope_type is {camera_pipe.telescope_type}")
            self.telescopes.append(telescope_type)
            self.camera_pipes[telescope_type] = camera_pipe
        elif isinstance(camerapipe_or_parameters, dict):
            camera_parameters = camerapipe_or_parameters
            camera_pipe = CameraPipe(telescope_type=telescope_type, version=version, **camera_parameters)
            self.telescopes.append(telescope_type)
            self.camera_pipes[telescope_type] = camera_pipe
        elif camerapipe_or_parameters is not None:
            raise ValueError(f"Invalid camerapipe_or_parameters")

    def __call__(self, cameras, telescopes):
        if not isinstance(cameras, list):
            cameras = [cameras]
        if not isinstance(telescopes, list):
            telescopes = [telescopes]

        # Build list with loaded images and indices
        respond = [None] * len(cameras)

        # Split by telescope
        camera_by_telescope = {t:[] for t in self.telescopes}
        indices_by_telescope = {t:[] for t in self.telescopes}
        try:
            for i, (camera, telescope) in enumerate(zip(cameras, telescopes)):
                camera_by_telescope[telescope].append(camera)
                indices_by_telescope[telescope].append(i)
        except KeyError as err:
            raise ValueError(f"MultiCameraPipe doesn't support {err.args[0]}")

        # Call CameraPipe by telescope
        for telescope, camera_pipe in self.camera_pipes.items():
            # Apply preprocessing
            cameras_t = camera_by_telescope[telescope]
            indices_t = indices_by_telescope[telescope]
            cameras_t_preprocessed = camera_pipe(cameras_t)
            # Copy results
            for i, c in zip(indices_t, cameras_t_preprocessed):
                respond[i] = c
        return respond


class TelescopeFeaturesPipe():
    def __init__(self, telescope_type=None, array_scaler_path=None, version="ML1"):
        self.telescope_type = telescope_type
        self.array_scaler = load_scaler(array_scaler_path)
        self.version = version

    def __call__(self, telescope_features):
        if self.array_scaler is not None: 
            if len(telescope_features.shape) == 1:
                telescope_features = telescope_features.reshape((1, -1))
            return self.array_scaler.transform(telescope_features)
