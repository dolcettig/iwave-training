
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, LinearNDInterpolator

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT.parent))
from common.utils import median_filter_iwave


# inputs
filename = "ex2R_vid" # video ID
bathymetry_path = ROOT / "data" / "GCPs" / "bathymetry_GPS.csv"
velocity_path = ROOT / "results" / f"{filename}_results_georeferenced.nc"
output_path = ROOT / "results" / f"{filename}_transect_restults.csv"

surface_elevation = 0
velocity_index = 0.85
ds_transect = 0.5      # m - resolution along transect
average_width = 2     # m - length of streamwise averaging segment
n_average = 9 # number of interpolation points along streamwise averaging segment
dq = 0.015


# load transect
if bathymetry_path.suffix.lower()==".xlsx":
    df = pd.read_excel(bathymetry_path)
else:
    df = pd.read_csv(bathymetry_path)

mask = np.isfinite(df["Easting"])
x = df.loc[mask,"Easting"].to_numpy()
y = df.loc[mask,"Northing"].to_numpy()
bed = df.loc[mask,"Elevation"].to_numpy()

if "Distance" in df.columns:
    s = df.loc[mask,"Distance"].to_numpy()
else:
    s = np.insert(np.cumsum(np.hypot(np.diff(x), np.diff(y))),0,0)

sreg = np.arange(s[0], s[-1], ds_transect)
sreg = np.append(sreg, s[-1])

fx = interp1d(s,x)
fy = interp1d(s,y)
fz = interp1d(s,bed)

x = fx(sreg)
y = fy(sreg)
bed = fz(sreg)
depth = np.maximum(surface_elevation-bed,0)

dx = np.gradient(x,sreg)
dy = np.gradient(y,sreg)
L = np.hypot(dx,dy)
tx = dx/L
ty = dy/L
nx = -ty
ny = tx

# load velocities
ds = xr.open_dataset(velocity_path)

vE = ds.vE.values
vN = ds.vN.values
q = ds.q.values

vE_filtered, vN_filtered = median_filter_iwave(vE, vN, q, dq = dq)

# transform to world coordinates
xm = ds.xm.values
ym = ds.ym.values

xs = ds.xs.values
ys = ds.ys.values


points = np.column_stack((ds.xs.values.ravel(),
                          ds.ys.values.ravel()))

vE_interp = LinearNDInterpolator(points, vE_filtered.ravel(), fill_value=np.nan)
vN_interp = LinearNDInterpolator(points, vN_filtered.ravel(), fill_value=np.nan)

# sample transect
vn = np.full_like(sreg, np.nan, dtype=float)
r = np.linspace(-average_width/2, average_width/2, n_average)

for i in range(len(sreg)):
    xx = x[i] + r*nx[i]
    yy = y[i] + r*ny[i]

    vxs = vE_interp(xx, yy)
    vys = vN_interp(xx, yy)

    comp = vxs*nx[i] + vys*ny[i]
    if np.any(np.isfinite(comp)):
        vn[i] = np.nanmean(comp)

valid = np.isfinite(vn) & np.isfinite(depth)
Q = velocity_index * np.trapezoid(vn[valid]*depth[valid], sreg[valid])

print(f"Discharge = {Q:.3f} m3/s")

out = pd.DataFrame({
    "s": sreg,
    "Easting": x,
    "Northing": y,
    "depth": depth,
    "vn": vn,
    "unit_discharge": velocity_index*vn*depth
})
out.to_csv(output_path, index=False)


speed = np.sqrt(vE**2 + vN**2)

fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

pcm = ax.pcolormesh(
    ds.xs.values,
    ds.ys.values,
    speed,
    shading="auto",
    cmap="turbo"
)

plt.colorbar(pcm, ax=ax, label="Velocity magnitude (m/s)")


ax.quiver(
    ds.xs.values,
    ds.ys.values,
    vE,
    vN,
    color="k",
    scale=20,
    width=0.002
)

# transect
ax.plot(x, y, "r-", lw=3, label="Transect")

valid = np.isfinite(vn)

q = ax.quiver(
    x[valid],
    y[valid],
    vn[valid] * nx[valid],
    vn[valid] * ny[valid],
    vn[valid],          # color
    cmap="Reds_r",
    scale=10,
    width=0.005,
    zorder=10,
)

plt.colorbar(q, ax=ax, label="Normal velocity (m/s)")

# show averaging width
for i in range(0, len(sreg), 20):

    x1 = x[i] - average_width/2 * nx[i]
    y1 = y[i] - average_width/2 * ny[i]

    x2 = x[i] + average_width/2 * nx[i]
    y2 = y[i] + average_width/2 * ny[i]

    ax.plot([x1, x2], [y1, y2], "w-", lw=1)

ax.set_aspect("equal")
ax.set_xlabel("Easting (m)")
ax.set_ylabel("Northing (m)")
ax.legend()

plt.show()


fig, ax = plt.subplots(3, 1, figsize=(10, 6), sharex=True,
                       constrained_layout=True)

ax[0].plot(sreg, depth, color="black")
ax[0].set_ylabel("Depth (m)")
ax[0].grid(True)

ax[1].plot(sreg, vn, color="tab:red")
ax[1].set_ylabel("Normal velocity\n(m/s)")
ax[1].grid(True)

unit_q = velocity_index * vn * depth

ax[2].plot(sreg, unit_q, color="tab:blue")
ax[2].fill_between(sreg, 0, unit_q,
                   color="tab:blue", alpha=0.3)
ax[2].set_ylabel("Unit discharge\n(m²/s)")
ax[2].grid(True)

cumQ = velocity_index * np.concatenate(
    ([0], np.cumsum(
        (unit_q[:-1] + unit_q[1:]) / 2 * np.diff(sreg)
    ))
)

ax[2].set_xlabel("Distance along transect (m)")

plt.show()