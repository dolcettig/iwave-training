### Make an orthovideo from the original 

import pyorc
import sys
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT.parent))
from common.utils import get_pyorc_cli


# INPUTS
filename = "ex2R_vid" # video ID
videopath = ROOT / "data" / "raw_video" / f"{filename}.mp4"
coordpath = ROOT / "data" / "GCPs" / "GCP_coordinates.xlsx"
output_path = ROOT / "data" / "rectified_video"

cam_config_name = "cam_config.json"


# GCPs coordinates file
df = pd.read_excel(coordpath)

# rectification parameters
fps = 25                        # video frame rate (fps)
crs = 25832                     # EPSG CRS zone
z_0 = 186.2                     # reference water surface level
h_ref = 0.0                     # water surface deviation from reference
target_res = 0.1 #0.05               # target image resolution (m/pxl)
correct_distortion = False
stabilise = False
rotation = 90 # correct camera rotation
duration = 10 # 30


# load pyorc
pyorc_exe = get_pyorc_cli()


# RECTIFICATION
output_path.mkdir(parents=True, exist_ok=True)

# load GCPs coordinates. Check naming
dst_list = df[["Easting", "Northing", "Height"]].values.tolist()

# remove elevation for nadir 2-points scaling
if len(dst_list) < 3:
    dst_list = df[["Easting", "Northing"]].values.tolist()

# create camera configuration file
cmd = [
            str(pyorc_exe), "camera-config",
            "-V", str(videopath),
            "--crs", "EPSG:"+str(crs),
            "--crs_gcps", "EPSG:"+str(crs),
            "--dst", str(dst_list),
            "--z_0", str(z_0),
            "--h_ref", str(h_ref),
            "--resolution", str(target_res),
            "--rotation", str(rotation),
            "--window_size", "64"              # the window size parameter is only relevant for pyorc velocimetry
        ]

        
# neglect camera distortion 
if not correct_distortion:
    cmd.extend(["--k1", "0.0", "--k2", "0.0"])
    
# apply stabilisation
if stabilise:
    cmd.extend(["--stabilize"])

cmd.extend([
    "-vvv",
    str(output_path / cam_config_name)
    ])

print(cmd)

subprocess.run(cmd, check=True)

cam_conf = pyorc.load_camera_config(output_path / cam_config_name)

t1 = time.time()
vid = pyorc.Video(videopath, camera_config=cam_conf, h_a=0.0, lazy=True, fps=fps, rotation=rotation,
                      start_frame=0, end_frame=fps*duration)
fr = vid.get_frames()


frame = vid.get_frame(0, method="rgb")


f_p = fr.frames.project(method="numpy")

# save rectified video
f_p.frames.to_video(output_path / f"{filename}_rectified.mp4")

# create and store coordinates  
y = f_p.y
x = f_p.x
ys = f_p.ys
xs = f_p.xs
lon = f_p.lon
lat = f_p.lat
Y,X =   np.meshgrid(y.values, x.values)
        
coord_df = pd.DataFrame({
    "x": X.flatten(),
    "y": Y.flatten(),
    "xs": xs.values.flatten(order='F'),
    "ys": ys.values.flatten(order='F'),
    "lon": lon.values.flatten(order='F'),
    "lat": lat.values.flatten(order='F')
})
coord_df.to_csv(output_path / f"{filename}_rectified.csv", index = False)


