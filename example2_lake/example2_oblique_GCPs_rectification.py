### Make an orthovideo from the original 

import pyorc
import sys
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT.parent))
from common.utils import get_pyorc_cli, stabilisation_polygon


# INPUTS
filename = "ex2L_vid" # video ID
videopath = ROOT / "data" / "raw_video" / f"{filename}.mp4"
coordpath = ROOT / "data" / "GCPs" / "GCP_coordinates.xlsx"
output_path = ROOT / "data" / "rectified_video"

cam_config_name = "cam_config.json"


# GCPs coordinates file
df = pd.read_excel(coordpath)

# rectification parameters
fps = 25                        # video frame rate (fps)
crs = 25832                     # EPSG CRS zone
z_0 = 0.0                     # reference water surface level
h_ref = 0.0                     # water surface deviation from reference
target_res = 0.1 #0.05               # target image resolution (m/pxl)
correct_distortion = False
stabilise = True
rotation = 0 # correct camera rotation
duration = 10 # 30


# load pyorc
pyorc_exe = get_pyorc_cli()


# RECTIFICATION
output_path.mkdir(parents=True, exist_ok=True)


# Build dst  and src list
dst_list = []
src_list = []
for _, row in df.iterrows():
    x = row["Easting"]
    y = row["Northing"]
    z = 0 #row.get("Altitude", Z_DEFAULT)
    
    pxl_x = row["pxl_x"]
    pxl_y = row["pxl_y"]
    
    dst_list.append([x, y, z])
    src_list.append([int(pxl_x), int(pxl_y)])
    
dst_str = str(dst_list)
src_str = str(src_list)

print(dst_str)
print(src_str)


if not correct_distortion:
    dist = np.zeros((5,1), dtype=float)

cap = cv2.VideoCapture(str(videopath))
ret, frame = cap.read()

if stabilise:
    vertices = stabilisation_polygon(frame)
    print("Selected polygon vertices:", vertices)


cam_config = pyorc.CameraConfig(
    height=frame.shape[0], 
    width=frame.shape[1], 
    resolution = target_res, 
    dist_coeffs = dist, 
    crs=crs, 
    window_size=128,
    stabilize=vertices if stabilise else None
    )

cam_config.set_gcps(src=src_list,  dst=dst_list, z_0=z_0, crs=crs, h_ref=0.0)

cam_config.calibrate()


plt.figure(figsize=(12, 7))
plt.imshow(frame)
plt.title(
    "Click 4 corner points in order:\n"
    "1. Upstream left-bank\n"
    "2. Downstream left-bank\n"
    "3. Downstream right-bank\n"
    "4. Upstream right-bank\n"
)
plt.axis("on")

# Click and record 4 points
points = plt.ginput(4, timeout=0)
plt.close()

corners = [[int(x), int(y)] for x, y in points]



cam_config.set_bbox_from_corners(corners)

print(cam_config)
print(corners)
print(frame.shape)

# ax = plt.axes()
cam_config.plot(
    tiles="GoogleTiles", 
    tiles_kwargs=dict(style="satellite"),
    zoom_level=18
    )
plt.savefig(output_path / "cam_config.png", dpi=300, bbox_inches='tight')
plt.show()

ax = plt.axes()
ax.imshow(frame)
plt.savefig(output_path / "orig_frame.png")
cam_config.plot(ax=ax, mode='camera')
plt.savefig(output_path / "orig_frame_GCPs.png")

cam_config.to_file(output_path / cam_config_name)


t1 = time.time()
vid = pyorc.Video(videopath, camera_config=cam_config, lazy=True, fps=fps, end_frame = fps*duration)
fr = vid.get_frames()



f_p = fr.frames.project(method="numpy")
f_p.frames.to_video(output_path / f"{filename}_rectified.mp4")

        # project coordinates
y = f_p.y
x = f_p.x
ys = f_p.ys
xs = f_p.xs
lon = f_p.lon
lat = f_p.lat
Y,X =   np.meshgrid(y.values, x.values)
        
df = pd.DataFrame({
    "x": X.flatten(),
    "y": Y.flatten(),
    "xs": xs.values.flatten(order='F'),
    "ys": ys.values.flatten(order='F'),
    "lon": lon.values.flatten(order='F'),
    "lat": lat.values.flatten(order='F')
})
df.to_csv(output_path / f"{filename}_rectified.csv", index = False)