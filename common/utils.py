import sys
import numpy as np
from scipy.ndimage import median_filter
from shutil import which
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector

def get_pyorc_cli():
    env = Path(sys.executable).parent

    if sys.platform == "win32":
        exe = env / "Scripts" / "pyorc.exe"
    else:
        exe = env / "pyorc"

    if not exe.exists():
        exe = which("pyorc")
        if exe is not None:
            pyorc_exe = Path(exe)
        else:
            raise RuntimeError(...)

    return exe


def median_filter_iwave(vx, vy, q, size=(3,3), dq=0.05, qth=0.6, verbose=False):
    """Filter out poor quality u,v results iteratively using local median of q."""
    
    # working copy of q that gets progressively masked
    q_work = q.copy()
    vx_work = vx.copy()
    vy_work = vy.copy()

    # keep original invalid points masked
    invalid0 = np.isnan(vx_work) | np.isnan(vy_work) | np.isnan(q_work)
    vx_work[invalid0] = np.nan
    vy_work[invalid0] = np.nan
    q_work[invalid0] = np.nan

    iteration = 0

    while True:
        valid = ~np.isnan(q_work)
        if not np.any(valid):
            break

        # temporarily replace NaNs with current mean q
        fill_value = np.nanmean(q_work)
        # fill_value = np.nanmin(q_work)
        q_fill = np.where(np.isnan(q_work), fill_value, q_work)

        # local q median
        q_med = median_filter(q_fill, size=size, mode="reflect")

        # identify bad vectors only among currently valid points
        bad = valid & (q_work < (q_med - dq))
        n_bad = np.sum(bad)

        if verbose:
            print(f"Iteration {iteration+1}: removing {n_bad} vectors")

        # stop when no new outliers are found
        if n_bad == 0:
            break

        # remove bad vectors
        vx_work[bad] = np.nan
        vy_work[bad] = np.nan
        q_work[bad] = np.nan

        iteration += 1
        
    mask_threshold = q_work >= qth
    vx_work[~mask_threshold] = np.nan
    vy_work[~mask_threshold] = np.nan

    return vx_work, vy_work


def stabilisation_polygon(img):

    state = {
        "vertices": None,
        "finished": False,
        "cancelled": False,
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(img)
    ax.set_title(
        "Draw the stabilisation exclusion polygon\n\n"
        "Left click: add points\n"
        "Right click: remove point\n"
        "Press ENTER when finished"
    )

    def onselect(verts):
        state["vertices"] = [
            [int(round(x)), int(round(y))]
            for x, y in verts
        ]

    selector = PolygonSelector(
        ax,
        onselect,
        useblit=True,
        props=dict(
                   color="red",
                   linewidth=2),
    )

    def on_key(event):
        if event.key == "enter":
            if state["vertices"] is None or len(state["vertices"]) < 3:
                print("Please select at least 3 points.")
                return

            state["finished"] = True
            plt.close(fig)

        elif event.key == "escape":
            state["cancelled"] = True
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.show(block=True)

    if state["cancelled"]:
        return None

    return state["vertices"]