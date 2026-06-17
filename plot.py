import os
import numpy as np
import imageio.v2 as imageio

os.makedirs("/tmp/mpmwarp_matplotlib_cache",exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR","/tmp/mpmwarp_matplotlib_cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pxr import Usd,UsdGeom

usd_path="outputs/line.usd"
mp4_path="outputs/line.mp4"
prim_path="/root/particles"
fps=30

stage=Usd.Stage.Open(usd_path)
if stage is None:
    raise RuntimeError(f"Failed to open {usd_path}")

prim=stage.GetPrimAtPath(prim_path)
if not prim:
    raise RuntimeError(f"Missing prim {prim_path} in {usd_path}")

instancer=UsdGeom.PointInstancer(prim)
pos_attr=instancer.GetPositionsAttr()
times=pos_attr.GetTimeSamples()
if not times:
    times=[Usd.TimeCode.Default()]

all_positions=[]
for time_code in times:
    positions=pos_attr.Get(time_code)
    if not positions:
        continue
    positions_np=np.array([[p[0],p[1],p[2]] for p in positions],dtype=np.float32)
    all_positions.append((time_code,positions_np))

if not all_positions:
    raise RuntimeError(f"No positions found in {prim_path}")

stacked=np.concatenate([positions for _,positions in all_positions],axis=0)
pad=0.5
xmin=float(stacked[:,0].min())-pad
xmax=float(stacked[:,0].max())+pad
ymin=float(stacked[:,1].min())-pad
ymax=float(stacked[:,1].max())+pad

os.makedirs(os.path.dirname(mp4_path),exist_ok=True)

writer=imageio.get_writer(mp4_path,fps=fps,codec="libx264",quality=8,macro_block_size=16)
try:
    for time_code,positions in all_positions:
        fig=plt.figure(figsize=(8,4),dpi=160)
        ax=fig.add_subplot(111)
        ax.plot(positions[:,0],positions[:,1],linewidth=1.5,color="#1f77b4")
        ax.scatter(positions[:,0],positions[:,1],s=10,color="#ff7f0e")
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(ymin,ymax)
        ax.set_aspect("equal",adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"{usd_path} time={time_code}")
        ax.grid(True,alpha=0.3)
        fig.tight_layout()
        fig.canvas.draw()

        frame=np.asarray(fig.canvas.buffer_rgba())[...,:3].copy()
        writer.append_data(frame)
        plt.close(fig)
finally:
    writer.close()

print(mp4_path)
