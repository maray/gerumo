"""
Data and image transformations and normalization
================================================

This module handle telescope data, event data and camera data to normalize
and apply different preprocessing function. This modele also can generate 
different square images from charge and peakpos value.
"""

import numpy as np
import tables
from ctapipe.instrument import CameraGeometry
from astropy import units
from os import path
from . import (
    IMAGES_SIZE, INPUT_SHAPE, 
    TELESCOPES, TELESCOPES_ALIAS,  TELESCOPE_CAMERA, TELESCOPES_INVERSE_ALIAS,
    PIXELS_POSITION
)
import threading

lock = threading.Lock()

def synchronized_open_file(*args, **kwargs):
    with lock:
        return tables.open_file(*args, **kwargs)

def synchronized_close_file(self, *args, **kwargs):
    with lock:
        return self.close(*args, **kwargs)

__all__ = [
    'load_camera', 'load_cameras', 'load_camera_geometry',
    'camera_to_image', 'cameras_to_images'
]


_images_attributes = {
    "ML1": {
        "charge":   "image_charge",
        "peakpos":   "image_peak_times",
    },
    "ML2": {
        "charge":   "charge",
        "peakpos":   "peakpos",
    },
    "DL1": {
        "charge":   "image",
        "peakpos":  "peak_time",
    }
}

#parameters obtained from dataset
CAM_PIX_AREA = {
    "LSTCam" : 0.002079326892271638,
    "NectarCam" : 0.002079326892271638
}


def load_camera(source, folder, telescope_type, observation_indice, version="ML1"):
    """Load charge and timepeak from hdf5 file for a observation."""

    hdf5_filepath = path.join(folder, source)
    telescope_alias = TELESCOPES_ALIAS[version][telescope_type]
    with tables.open_file(hdf5_filepath, "r") as hdf5_file:
        image = hdf5_file.root[telescope_alias][observation_indice]
    charge = image[_images_attributes[version]["charge"]]
    peakpos = image[_images_attributes[version]["peakpos"]]
    return charge, peakpos

def load_cameras(dataset, version="ML1"):
    """Load charge and time peak from hdf5 files for a dataset.
    
    Returns
    =======
        `list` of `tuples` of (`np.ndarray`, `np.ndarray`)
        A list with the charge and peakpos values for each camera observations.
    """
    # avaliable files and telescopes 
    hdf5_filepaths = dataset["hdf5_filepath"].unique()
    telescopes = dataset["type"].unique()
    # build list with loaded images
    respond = [None] * len(dataset)
    indices = np.arange(len(dataset))
    # iterate over file
    if version == "DL1":
        tel_ids = dataset["telescope_id"].unique()
        for hdf5_filepath in hdf5_filepaths:
            #with tables.open_file(hdf5_filepath, "r") as hdf5_file:
            hdf5_file = synchronized_open_file(hdf5_filepath, mode="r")
            # and over telescope tables
            for tel_id in tel_ids:
                # select indices for this file and telescope
                selector = (dataset["hdf5_filepath"] == hdf5_filepath) & (dataset["telescope_id"]==tel_id)# & (dataset["type"] == telescope_type) #the condition is removed becasue it is asumed the dataset is laready filtered by type
                respond_indices_selected = indices[selector]
                observations_indices_selected = dataset[selector]["image_index"]#.to_numpy()
                image_data_table = hdf5_file.root.dl1.event.telescope.images[f'tel_{tel_id:03d}']
                images = image_data_table[observations_indices_selected.values]
                for i, img in zip(respond_indices_selected, images):
                    respond[i] = (img[_images_attributes[version]["charge"]], img[_images_attributes[version]["peakpos"]]) 
            synchronized_close_file(hdf5_file)

#                observations_indices_selected = np.array([list(map(int, row.split("_"))) for row in observations_indices_selected])
#                respond_indices_selected = indices[selector]
                # load images and copy results
#                for i, obs_ind_data in zip(respond_indices_selected,observations_indices_selected):
#                    obs_id = obs_ind_data[0]
#                    event_id = obs_ind_data[1]
#                    tel = obs_ind_data[2]
#                    tel_name = f'tel_{tel:03d}'
#                    tel_tabla = hdf5_file.root.dl1.event.telescope.images[tel_name]
#                    for img in tel_tabla:
#                        if img["obs_id"] == obs_id and img["event_id"] == event_id:
#                            break
#                    respond[i] = (img[_images_attributes[version]["charge"]], img[_images_attributes[version]["peakpos"]])
#            synchronized_close_file(hdf5_file)

    else:
        for hdf5_filepath in hdf5_filepaths:
            #with tables.open_file(hdf5_filepath, "r") as hdf5_file:
            hdf5_file = synchronized_open_file(hdf5_filepath, mode="r")
            # and over telescope tables
            for telescope_type in telescopes:
                telescope_alias = TELESCOPES_ALIAS[version][telescope_type]
                # select indices for this file and telescope
                selector = (dataset["hdf5_filepath"] == hdf5_filepath) & (dataset["type"] == telescope_type)
                observations_indices_selected = dataset[selector]["observation_indice"].to_numpy()
                respond_indices_selected = indices[selector]
                # load images and copy results
                images = hdf5_file.root[telescope_alias][observations_indices_selected]
                for i, img in zip(respond_indices_selected, images):
                    respond[i] = (img[_images_attributes[version]["charge"]], img[_images_attributes[version]["peakpos"]]) 
            # Free lock
            synchronized_close_file(hdf5_file)
    return respond
    
def load_camera_geometry(telescope_type, version="ML1"):
    pixpos = PIXELS_POSITION[version]['raw'][telescope_type]
    n_pix = len(pixpos[0])
    geometry = CameraGeometry(
        name= TELESCOPE_CAMERA[telescope_type],
        pix_id= np.arange(n_pix),
        pix_x = units.quantity.Quantity(pixpos[0], 'meter'),
        pix_y = units.quantity.Quantity(pixpos[1], 'meter'),
        pix_area= units.quantity.Quantity(np.full(n_pix,CAM_PIX_AREA[TELESCOPE_CAMERA[telescope_type]]),'m2'),
        pix_type= "hexagonal"
    )
    return geometry

"""
Input preprocessing Functions
=============================

Prepare input images, normalize and standarize values
from images and features. 
"""

def _simple(charge, peakpos, telescope_type, mask, version="ML1"):
    """
    Transform charge and peak positions from raw format to square matrix.
        
    Parameters
    ----------
    charge : `numpy.ndarray`
        'charge' pixel values from the observation.
    peakpos : `numpy.ndarray`
        'peakpos' pixel values from the observation.
    telescope_type : `str`
        Telescope type.
    mask : `bool`, optional
        Add a mask channel to the result_image. (default=True)
    version : `str`, optional
        Prod3b version: ('ML1', 'ML2').
        This define the h5 structure and pixpos. (default='ML1')

    Returns
    =======
        `np.ndarray`
        Charge and peakpos values in square matrix form.
    """
    x, y = PIXELS_POSITION[version]["simple"][telescope_type] #(x, y)
    image_size = IMAGES_SIZE[telescope_type]
    if mask:
        input_shape = INPUT_SHAPE["simple-mask"][telescope_type]
    else:
        input_shape = INPUT_SHAPE["simple"][telescope_type]
    canvas = np.zeros(input_shape, dtype="float32")
    canvas[y, x, 0] = charge
    canvas[y, x, 1] = peakpos
    if mask:
        canvas[y, x, 2] = 1
    return canvas

def _simple_shift(charge, peakpos, telescope_type, mask, version="ML1"):
    """
    Transform charge and peak positions from raw format to two square matrices (left-right shifted).
        
    Parameters
    ----------
    charge : `numpy.ndarray`
        'charge' pixel values from the observation.
    peakpos : `numpy.ndarray`
        'peakpos' pixel values from the observation.
    telescope_type : `str`
        Telescope type.
    mask : `bool`, optional
        Add a mask channel to the result_image. (default=True)
    version : `str`, optional
        Prod3b version: ('ML1', 'ML2').
        This define the h5 structure and pixpos. (default='ML1')

    Returns
    =======
        `np.ndarray`
        Charge and peakpos values in square matrix form.
    """
    x_left, x_right, y = PIXELS_POSITION[version]["simple_shift"][telescope_type] #(x_l, x_r, y)
    image_size = IMAGES_SIZE[telescope_type]
    if mask:
        input_shape = INPUT_SHAPE["simple-shift-mask"][telescope_type]
    else:
        input_shape = INPUT_SHAPE["simple-shift"][telescope_type]
    canvas = np.zeros(input_shape, dtype="float32")
    canvas[0, y, x_left, 0] = charge
    canvas[0, y, x_left, 1] = peakpos
    canvas[1, y, x_right, 0] = charge
    canvas[1, y, x_right, 1] = peakpos
    if mask:
        canvas[0, y, x_left, 2] = 1
        canvas[1, y, x_right, 2] = 1
    return canvas

def _time(peakpos, telescope_type, mask, version="ML1"):
    """
    Transform only peak positions from raw format to one square matrix.
    
    Parameters
    ----------
    peakpos : `numpy.ndarray`
        'peakpos' pixel values from the observation.
    telescope_type : `str`
        Telescope type.
    mask : `bool`, optional
        Add a mask channel to the result_image. (default=True)
    version : `str`, optional
        Prod3b version: ('ML1', 'ML2').
        This define the h5 structure and pixpos. (default='ML1')

    Returns
    =======
        `np.ndarray`
        Peakpos values in square matrix form.
    """
    x, y = PIXELS_POSITION[version]["time"][telescope_type] #(x, y)
    image_size = IMAGES_SIZE[telescope_type]
    if mask:
        input_shape = INPUT_SHAPE["time-mask"][telescope_type]
    else:
        input_shape = INPUT_SHAPE["time"][telescope_type]
    canvas = np.zeros(input_shape, dtype="float32")
    canvas[y, x, 0] = peakpos
    if mask:
        canvas[y, x, 1] = 1
    return canvas

def _time_shift(peakpos, telescope_type, mask, version="ML1"):
    """
    Transform only peak positions from raw format to two square matrices (left-right shifted).
        
    Parameters
    ----------
    peakpos : `numpy.ndarray`
        'peakpos' pixel values from the observation.
    telescope_type : `str`
        Telescope type.
    mask : `bool`, optional
        Add a mask channel to the result_image. (default=True)
    version : `str`, optional
        Prod3b version: ('ML1', 'ML2').
        This define the h5 structure and pixpos. (default='ML1')

    Returns
    =======
        `np.ndarray`
        Peakpos values in square matrix form.
    """
    x_left, x_right, y = PIXELS_POSITION[version]["time_shift"][telescope_type] #(x_l, x_r, y)
    image_size = IMAGES_SIZE[telescope_type]
    if mask:
        input_shape = INPUT_SHAPE["time-shift-mask"][telescope_type]
    else:
        input_shape = INPUT_SHAPE["time-shift"][telescope_type]
    canvas = np.zeros(input_shape, dtype="float32")
    canvas[0, y, x_left, 0] = peakpos
    canvas[1, y, x_right, 0] = peakpos
    if mask:
        canvas[0, y, x_left, 1] = 1
        canvas[1, y, x_right, 1] = 1
    return canvas


def camera_to_image(charge, peakpos, telescope_type, mode="simple", mask=True, version="ML1"):
    """
    Transform the charge and peakpos values into an rectangle image.

    The transformation can be done in different modes:
    - 'simple' just return an image two channels (charge and peakpos), 
    where each pixel is vertical aligned. 
    - 'simple-shift' create two version of each image, where odd rows are 
    shifted to right and left, respectively.

    An extra channel can be added with the flag 'mask', this mask represent which pixels
    are data from the camera.

    Parameters
    ----------
    charge : `numpy.ndarray`
        'charge' pixel values from the observation.
    peakpos : `numpy.ndarray`
        'peakpos' pixel values from the observation.
    telescope_type : `str`
        Telescope type.
    mode : `str`, optional
        Image transformation mode: ('simple', 'simple-shift').
        This define the result_image. (default='simple')
    mask : `bool`, optional
        Add a mask channel to the result_image. (default=True)
    Returns
    -------
    numpy.array or tuple of numpy.array
        result_image
    """
    if mode == "simple":
        result_image = _simple(charge, peakpos, telescope_type, mask, version)
    elif mode == "simple-shift":
        result_image = _simple_shift(charge, peakpos, telescope_type, mask, version)
    elif mode == "time":
        result_image = _time(peakpos, telescope_type, mask, version)
    elif mode == "time-shift":
        result_image = _time_shift(peakpos, telescope_type, mask, version)
    elif mode == "raw":
        result_image = np.vstack((charge, peakpos))
    else:
        raise ValueError(f"invalid 'mode': {mode}")
    return result_image

def cameras_to_images(cameras, telescopes_type, mode="simple", mask=True, version="ML1"):
    """
    Transform the charge and peakpos values into rectangle images.

    The transformation can be done in different modes:
    - 'simple' just return an image two channels (charge and peakpos), 
    where each pixel is vertical aligned. 
    - 'simple-shift' create two version of each image, where odd rows are 
    shifted to right and left, respectively.
    - 'split-time' mode generates just one multichannel image, where the first channel
    is the charge, and the other channels have binary values and represent each value 
    in peakpos, like a one-hot representation. 
    - 'split-time-shift' create two shifted version of the 'split-time' mode.

    An extra channel can be added with the flag 'mask', this mask represent which pixels
    are data from the camera.

    Parameters
    ----------
    cameras : `list` of `tuple` of `numpy.ndarray`
        Containt 'charge' and 'peakpos' pixel values for each observation.
    telescope_type : `list` of `str`
        List of telescope type for each observation.
    mode : `str`, optional
        Image transformation mode: ('simple', 'simple-shift').
        This define the result_image. (default='simple')
    mask : `bool`, optional
        Add a mask channel to the result_image. (default=True)
    version : `str`, optional
        Prod3b ML version. (default='ML1')
    Returns
    -------
    `list` of `numpy.ndarray` or `list` of `tuple` of `numpy.ndarray`
        result_images
    """
    result_images = []
    for (charge, peakpos), telescope_type in zip(cameras, telescopes_type):
        if version == "DL1":
            telescope_type = TELESCOPES_INVERSE_ALIAS[telescope_type]
        result_images.append(camera_to_image(charge, peakpos, telescope_type, mode, mask, version))
    return result_images
