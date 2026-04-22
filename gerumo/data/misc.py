"""
Utilities functions
===================
"""

from math import ceil
from os.path import join
import numpy as np
import tables
from .dataset import _telescope_table, _telescopes_info_attributes
from . import TELESCOPES, TELESCOPES_ALIAS

__all__ = [
    'get_resolution', 'get_shape', 
    'extract_pixel_positions',
    'estimate_alphas', 'get_alphas'
]


"""
Misc Functions
===============
"""

def get_resolution(targets, targets_domain, targets_shape):
    """Return the targets resolution for each target given the targets shape"""
    targets_resolution = {}
    for target in targets:
        vmin, vmax = targets_domain[target]
        shape = targets_shape[target]
        targets_resolution[target]  = (vmax -vmin) / shape
    return targets_resolution


def get_shape(targets, targets_domain, targets_resolution):
    """Return the targets shape for each target given the targets resolution"""
    targets_shape = {}
    for target in targets:
        vmin, vmax = targets_domain[target]
        resolution = targets_resolution[target]
        targets_shape[target]  = ceil((vmax -vmin) / resolution)
    return targets_resolution


"""
Pixel Positions Functions
===============
"""

def LST_LSTCam_align(pixels_position_array):
    xs = pixels_position_array[0,:]
    ys = pixels_position_array[1,:]
    # Distance matrix:
    delta_x = np.array([xs])-np.array([xs]).T
    delta_y = np.array([ys])-np.array([ys]).T
    dists = (delta_x**2+delta_y**2)**0.5
    angles = np.arctan2(delta_y, delta_x) # Angles from -pi to pi
    # Binary search, find maximum radious where no cell has more than 6 neighbors
    rad1 = 0
    rad2 = np.max(dists)
    for i in range(1000):
        rad = (rad1+rad2)/2.0
        neighs = dists<rad # matrix with true if i,j are neighbors
        np.fill_diagonal(neighs,False)
        max_neighs = np.max(np.sum(neighs,axis=1))
        if max_neighs>6:
            rad2 = rad
        else:
            rad1 = rad
    #
    rad = rad1
    neighs = dists<rad
    # Get a group of angles on an interval:
    ang_start = 0
    ang_end = np.pi*(6//2)
    # Neighbors with angle between those two
    conditions = np.all([neighs,angles>=ang_start,angles<ang_end],axis=0)
    neighbors = np.where(conditions)
    neigh_angles = angles[neighbors]
    # From the angles in this group, pick the median as the main axis
    main_axis_ang = np.median(neigh_angles)
    main_x = np.cos(main_axis_ang)
    main_y = np.sin(main_axis_ang)
    # Apply transformation
    tx = xs*main_x+ys*main_y
    ty = xs*main_y-ys*main_x
    # Now compute the maximum separation between neighboors in the main axis.
    dx = np.max(delta_x[neighs]*main_x+delta_y[neighs]*main_y)
    # Scale main axis by half of that separation:
    tx = np.round(tx/(dx/2.0))
    # Now compute the maximum separation between neighboors in the secondary axis.
    dy = np.max(delta_x[neighs]*main_y-delta_y[neighs]*main_x)
    # Scale secondary axis by that separation:
    ty = np.round(ty/dy)
    return np.stack((tx, ty))


def to_simple_and_shift(pixels_position_array):
    # get pixels positions
    xs = pixels_position_array[0]
    ys = pixels_position_array[1]
    # indices of x and y pixels position
    i = np.arange(0, len(ys))
    # row values of the telescope
    y_levels = np.sort(np.unique(ys))
    # image dimension
    nrows = len(y_levels)
    ncols = len(np.unique(xs))//2 + 1
    # new translated pixel positions
    new_x_l = np.copy(xs) # new pixels x positions left shift
    new_x_r = np.copy(xs) # new pixels x positions right shift
    new_y = np.copy(ys)
    # shift odd rows
    dx = 0
    for level, y_value in enumerate(y_levels):
        indices = i[ys == y_value]
        if dx == 0:
            dx = np.diff(np.sort(xs[indices])).min()/2
        if level % 2 != 0:
            new_x_l[indices] -= dx
            new_x_r[indices] += dx
    # round values
    new_x_l = np.round(new_x_l, 3)
    new_x_r = np.round(new_x_r, 3)
    # max indices of image output
    max_col_l = len(np.unique(new_x_l)) - 1
    max_col_r = len(np.unique(new_x_r)) - 1
    max_row = nrows - 1
    # apply linear transfomation
    new_x_l = ((max_col_l/(new_x_l.max() - new_x_l.min())) * (new_x_l - new_x_l.min()))
    new_x_l = np.round(new_x_l).astype(int)
    new_x_r = ((max_col_r/(new_x_r.max() - new_x_r.min())) * (new_x_r - new_x_r.min()))
    new_x_r = np.round(new_x_r).astype(int)
    new_y = ((max_row/(new_y.max() - new_y.min())) * (new_y - new_y.min()))
    new_y = np.round(new_y).astype(int)
    # prepare output
    simple = np.vstack((new_x_r, new_y))
    simple_shift = np.vstack((new_x_l, new_x_r, new_y))
    return simple, simple_shift


def extract_pixel_positions(hdf5_filepath, pixpos_folder, version="ML2"):
    """Extract and save from file the information about pixel position.
    Extract and apply transformation  to pixel position of each telescope 
    type and for each camera_to_image 'mode'. Saves it to use them in the
    'camera_to_image' function.

    Note: This function is used just once. The pixel position can be shared
    in a numpy file format.
    """
    inverse_alias = {TELESCOPES_ALIAS[version][t]:t for t in TELESCOPES}
    modes = ('raw', 'simple', 'simple-shift', 'time', 'time_shift')
    pixpos_folder = join(pixpos_folder, version)
    print(pixpos_folder)

    hdf5_file = tables.open_file(hdf5_filepath, "r")
    if version == "DL1":
        type_geometry_dict = {}
        telescopes_info = hdf5_file.root.configuration.instrument.subarray.layout
        geometry_group = hdf5_file.root.configuration.instrument.telescope.camera
    else:
        telescopes_info = hdf5_file.root[_telescope_table[version]]

    all_pixpos = {}
    
    # Extract pixel position array from hdf5 file
    raw_pixpos = {}
    if version == "DL1":
        for tel in telescopes_info:
            tipo = tel["type"].decode("utf-8")
            if tipo not in type_geometry_dict:
                type_geometry_dict[tipo] = tel["camera_index"]
        for t in type_geometry_dict:
            if t not in inverse_alias:
                continue
            geometry_index = "geometry_" + str(type_geometry_dict[t])
            raw_pixpos[t] = []
            geometry = geometry_group[geometry_index]
            num_pixels = len(geometry)
            raw_pixpos[t].append(geometry.cols.pix_x[:])
            raw_pixpos[t].append(geometry.cols.pix_y[:])
    else:
        for telescope in telescopes_info:
            type_ = telescope[_telescopes_info_attributes[version]["type"]].decode("utf-8")
            if type_ not in inverse_alias:
                continue
            num_pixels = telescope[_telescopes_info_attributes[version]["num_pixels"]]
            type_ = inverse_alias[type_]
            if version == "ML2":
                raw_pixpos[type_] = telescope[_telescopes_info_attributes[version]["pixel_pos"]][:num_pixels, :].T
            else:
                raw_pixpos[type_] = telescope[_telescopes_info_attributes[version]["pixel_pos"]][:, :num_pixels]
    
    # Save raw pixpos
    all_pixpos['raw'] = {}
    for telescope, pixpos in raw_pixpos.items():
        if telescope == TELESCOPES_ALIAS[version]["LST_LSTCam"]:
            LST_LSTCam_not_aligm = pixpos
            all_pixpos['raw']['LST_LSTCam_not_aligm'] = LST_LSTCam_not_aligm
            raw_pixpos["LST_LSTCam"] = LST_LSTCam_align(pixpos)
            pixpos = raw_pixpos["LST_LSTCam"]
            np.savetxt(join(pixpos_folder,'raw', f'{telescope}_not_align.npy'), LST_LSTCam_not_aligm)
        if telescope == TELESCOPES_ALIAS[version]["MST_NectarCam"]:
            MST_FlashCam_not_aligm = pixpos
            all_pixpos['raw']['MST_FlashCam_not_aligm'] = MST_FlashCam_not_aligm
            raw_pixpos[telescope] = LST_LSTCam_align(pixpos)
            pixpos = raw_pixpos[telescope]
            np.savetxt(join(pixpos_folder,'raw', f'{telescope}_not_align.npy'), MST_FlashCam_not_aligm)
        np.savetxt(join(pixpos_folder,'raw', f'{telescope}.npy'), pixpos)
        all_pixpos['raw'][telescope] = pixpos

    # Generate simple align and shift align
    all_pixpos['simple'] = {}
    all_pixpos['simple_shift'] = {}
    all_pixpos["time"] = {}
    all_pixpos["time_shift"] = {}
    for telescope, pixpos in raw_pixpos.items():
        simple, shift = to_simple_and_shift(pixpos)
        all_pixpos['simple'][telescope] = simple
        all_pixpos['time'][telescope] = simple
        all_pixpos['shift'][telescope] = shift
        all_pixpos['time_shift'][telescope] = shift
        np.savetxt(join(pixpos_folder,'simple', f'{telescope}.npy'), simple)
        np.savetxt(join(pixpos_folder,'time', f'{telescope}.npy'), simple)
        np.savetxt(join(pixpos_folder,'simple_shift', f'{telescope}.npy'), shift)
        np.savetxt(join(pixpos_folder,'time_shift', f'{telescope}.npy'), shift)
    
    return all_pixpos


"""
Focal Loss
==========
"""

def estimate_alphas(dataset, column="log10_mc_energy", bins=81, rescale=None):
    """
    Estimate alpha values for focal loss or weighted loss functions.

    Parameters
    ==========
    dataset :  `pd.DataFrame`
        Loaded dataset. Dataset of the events CSV with mc_energy transformed to log10
    column : `str`
        Target name, dataset column.
    bins : `int`
        Number of bins or classes.
    rescale : (`int`, `int`) or `None`
        Rescale value between given range.
    Returns
    -------
        `np.ndarray`
            Bins' weights (alpha values).
    """
    count, bins = np.histogram(dataset[column], bins=bins)
    count = count / (count.sum())
    alphas = (1 - count)
    if rescale is not None:
        min_, max_ = rescale
        alphas = (alphas - alphas.min()) * (max_ - min_) / (alphas.max() - alphas.min()) + min_
    else:
        alphas /= alphas.sum()
    return alphas


def get_alphas(telescope, version):
    "Alpha values for focal loss precomputed for log10 mc_energy with 81 bins in range (0.1, 1)"
    return {"ML1": {
        'SST1M_DigiCam': np.array(
               [0.99981382, 1.        , 1.        , 1.        , 0.99981382,
                0.99962764, 0.99832437, 0.99795201, 0.99664874, 0.99292511,
                0.9910633 , 0.98305751, 0.97300372, 0.94898635, 0.91696318,
                0.87190732, 0.79948283, 0.71030203, 0.61702524, 0.50717832,
                0.36623914, 0.30554406, 0.23758792, 0.22809268, 0.21245345,
                0.12736864, 0.11210178, 0.13425734, 0.14635912, 0.1       ,
                0.155482  , 0.24205627, 0.16423252, 0.17912702, 0.18043029,
                0.26030203, 0.33998759, 0.32658254, 0.3546959 , 0.32341746,
                0.35190319, 0.42525859, 0.50885395, 0.37964419, 0.55111709,
                0.48893256, 0.59580058, 0.5747621 , 0.63117501, 0.50680596,
                0.68386429, 0.67772031, 0.69950352, 0.7125362 , 0.6749276 ,
                0.66133637, 0.72612743, 0.75740588, 0.87991312, 0.73953248,
                0.78831196, 0.84155978, 0.83727762, 0.86576334, 0.87470004,
                0.84453868, 0.87637567, 0.89182871, 0.75666115, 0.86613571,
                0.92813405, 0.94582127, 0.95624741, 0.93576748, 0.9731899 ,
                0.96685974, 0.95810923, 0.94377327, 0.96779065, 0.95326851,
                0.98473314]),
        'MST_FlashCam': np.array(
               [1.        , 1.        , 0.99969724, 0.99969724, 0.99798161,
                0.99646782, 0.99576138, 0.98980713, 0.98143081, 0.96447634,
                0.93026463, 0.89554833, 0.83388652, 0.71782911, 0.62246019,
                0.47340211, 0.3558309 , 0.2185804 , 0.18608432, 0.10787172,
                0.10262391, 0.11745907, 0.10655977, 0.115037  , 0.1       ,
                0.10131195, 0.14672572, 0.12078941, 0.11231218, 0.16034985,
                0.24058085, 0.26742543, 0.27408612, 0.29982059, 0.36329895,
                0.38953801, 0.43131868, 0.44978695, 0.5142745 , 0.47400763,
                0.52366001, 0.58633102, 0.66706661, 0.61226732, 0.66111236,
                0.6804889 , 0.70753532, 0.75849966, 0.73145324, 0.7912985 ,
                0.76879345, 0.83681319, 0.82510653, 0.78655528, 0.84831801,
                0.87122673, 0.87082305, 0.88495178, 0.85053824, 0.87919937,
                0.91704418, 0.95670554, 0.9217874 , 0.93359498, 0.93026463,
                0.9565037 , 0.95034761, 0.95640278, 0.96750392, 0.93773268,
                0.92108096, 0.96074232, 0.98768782, 0.9880915 , 0.98052254,
                0.99515586, 0.98768782, 0.98435748, 0.98223817, 0.98455932,
                0.99555954]),
        'LST_LSTCam': np.array(
               [0.99395071, 0.98924571, 0.98386856, 0.95899925, 0.93244959,
                0.88640777, 0.83162808, 0.74861837, 0.64846901, 0.56445108,
                0.4942121 , 0.38599701, 0.34600448, 0.25056012, 0.22266617,
                0.2401419 , 0.12352502, 0.1       , 0.14301718, 0.21863331,
                0.19746079, 0.20888723, 0.27744586, 0.28752801, 0.34432412,
                0.37793129, 0.36616878, 0.37490665, 0.43338312, 0.50093353,
                0.48480209, 0.52076176, 0.54663928, 0.61620612, 0.60746826,
                0.6165422 , 0.65989544, 0.70056012, 0.69047797, 0.7529873 ,
                0.75097087, 0.75399552, 0.75466766, 0.78961912, 0.82389843,
                0.82961165, 0.84473488, 0.86456311, 0.8729649 , 0.89111277,
                0.89850635, 0.92270351, 0.90354742, 0.93547423, 0.93043316,
                0.91430172, 0.93749066, 0.94522031, 0.95026139, 0.95698282,
                0.94085138, 0.96504854, 0.97815534, 0.97647498, 0.97781927,
                0.98017177, 0.98487677, 0.9821882 , 0.98857356, 0.98924571,
                0.98151606, 0.96840926, 0.99058999, 0.99395071, 0.9976475 ,
                0.99428678, 1.        , 0.99932786, 0.99831964, 0.995295  ,
                0.99395071])
        },
        "DL1" : {
            'MST_NectarCam' : np.array(
                [0.99999506, 1.     ,    0.99997038, 0.99990128, 0.99984698, 0.99965448,
                0.99928428, 0.9984106,  0.99733455, 0.99441736, 0.98911113, 0.97966358,
                0.96351785, 0.93673498, 0.89874241, 0.84246187, 0.77139794, 0.68445975,
                0.58732868, 0.49292723, 0.406532,   0.33290628, 0.27234127, 0.21821283,
                0.18963325, 0.16311693, 0.139271,   0.12349547, 0.10933402, 0.1,
                0.10161902, 0.11152562, 0.11705396, 0.14202037, 0.16029353, 0.19898208,
                0.2243878,  0.25670888, 0.30321609, 0.35421015, 0.3925828,  0.43340865,
                0.47326211, 0.51376712, 0.54894122, 0.58348845, 0.61294664, 0.64687687,
                0.68256926, 0.70142487, 0.73144576, 0.76105203, 0.77250361, 0.79293381,
                0.82017079, 0.83034393, 0.85140101, 0.8702813,  0.87884036, 0.89234039,
                0.90352048, 0.90782963, 0.91464134, 0.93184832, 0.93096477, 0.93752968,
                0.94987468, 0.95305348, 0.96157306, 0.96222955, 0.96913998, 0.96992974,
                0.97392792, 0.97543341, 0.98235865, 0.98415043, 0.9825709,  0.99130766,
                0.99208262, 0.99309944, 0.99375593]),
            'LST_LSTCam' : np.array(
                [1.        , 0.99895556, 0.99754335, 0.99485134, 0.98954087, 0.98106765,
                0.96791651, 0.94623331, 0.9097661 , 0.86676746, 0.80058515, 0.73221098,
                0.64865563, 0.57348523, 0.49269544, 0.43751655, 0.37383501, 0.33585263,
                0.29923833, 0.26712541, 0.23362972, 0.20384106, 0.19345548, 0.15488469,
                0.14839738, 0.14451382, 0.11757899, 0.10832611, 0.10607542, 0.12087413,
                0.1       , 0.14099802, 0.15637044, 0.18252562, 0.21150521, 0.25532763,
                0.26972916, 0.29682581, 0.34457593, 0.38981056, 0.43241202, 0.46392181,
                0.49781468, 0.54487341, 0.57719227, 0.60177343, 0.63540151, 0.66314542,
                0.70117193, 0.72122227, 0.74346447, 0.77597457, 0.78534512, 0.80501299,
                0.82532812, 0.84202448, 0.8616188 , 0.88184567, 0.89073078, 0.90520586,
                0.91622399, 0.92109315, 0.92265246, 0.9389075 , 0.94159952, 0.94857227,
                0.95838414, 0.96097318, 0.96681323, 0.96862261, 0.97513934, 0.97559536,
                0.97822854, 0.98042039, 0.98558376, 0.98839346, 0.98624573, 0.99333617,
                0.99357153, 0.99495432, 0.99545447])
        }}[version][telescope]