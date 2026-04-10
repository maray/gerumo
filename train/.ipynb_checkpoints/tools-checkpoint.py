from os.path import basename, join, splitext
import cv2
import numpy as np
import pandas as pd
import glob
import argparse

ap = argparse.ArgumentParser(description="Create videos from experiments folder.")
ap.add_argument("-i", "--experiment", required=False, help="Path to experiment folder.", default='../alt_az/UMONNA_UNIT_LST_V2_437e0b', type=str)
ap.add_argument("-o", "--output", required=False, help="Output folder", default="./.", type=str)
args = vars(ap.parse_args())
experiment_folder = args["experiment"]
experiment_name = basename(experiment_folder)
output_folder = args["output"]

# Train plot
train_loss_filepath = glob.glob(f'{experiment_folder}/*Training Loss.png') 
train_loss_img = cv2.imread(f"{train_loss_filepath[0]}") if len(train_loss_filepath) == 1 else None

# Checkpoints 
checkpoints_models = glob.glob(f'{experiment_folder}/checkpoints/*.h5')
epochs = [{"epoch": float(epoch[0][1:]) - 1, "loss": float(epoch[1])} for epoch in map(lambda s:splitext(basename(s))[0].split("_")[-2:], checkpoints_models)] 
checkpoints_data = pd.DataFrame(epochs)
checkpoints_data["models"] = checkpoints_models
if len(checkpoints_data) > 0:
    checkpoints_data = checkpoints_data.sort_values("epoch")
else:
    checkpoints_data = None

# Regresion
regression_plots = glob.glob(f'{experiment_folder}/regression/*.png')
epochs = [dict(zip(epoch[::2], map(float, epoch[1::2]))) for epoch in map(lambda s:splitext(basename(s))[0].split("_"), regression_plots)]
regression_data = pd.DataFrame(epochs)
regression_data["plots"] = regression_plots
if len(checkpoints_data) > 0:
    regression_data = regression_data.sort_values("epoch")
else:
    regression_data = None

# Draw Video
font = cv2.FONT_HERSHEY_SIMPLEX 
fps = 2
img_array = []
video_filename = f'{experiment_name}.avi'
size = (0, 0)
## Traning
if train_loss_img is not None:
    height, width, layers = train_loss_img.shape
    height, width = max(height, size[1]), max(width, size[0])
    size = (width,height)
    img_array.extend(2*fps*[train_loss_img])

## Regression
for _, row in regression_data.iterrows():
    epoch = row["epoch"]
    loss = row["loss"]
    filename = row["plots"]
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    label = f"EPOCH:    {int(epoch)}"
    cv2.putText(img, label, (50, 50), font, 1,  
                (0, 0, 0),  4, cv2.LINE_4) 
    label = f"LOSS:     {loss}"
    cv2.putText(img, label, (50, 100), font, 1,  
                (0, 0, 0),  4, cv2.LINE_4)
    img_array.append(img)
    img_array.extend(1*fps*[img])

# Render video
video = cv2.VideoWriter(join(output_folder, video_filename), cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
for img in img_array:
    height, width, layers = img.shape
    if (size) != (width, height):
        # add padding
        new_width, new_height = size
        canvas = 255*np.ones((new_height, new_width, layers), dtype="uint8")
        # compute center offset
        xx = (new_width - width) // 2
        yy = (new_height - height) // 2
        canvas[yy:yy+height, xx:xx+width] = img
        img = canvas.copy()
    video.write(img)
video.release()