import os
import tempfile
from concurrent.futures import ProcessPoolExecutor

os.makedirs("/tmp/mpmwarp_matplotlib_cache",exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR","/tmp/mpmwarp_matplotlib_cache")

import imageio.v2 as imageio
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pxr import Usd,UsdGeom

import generate

DEFAULT_VIDEO_PATH="outputs/videos/mpm.mp4"
DEFAULT_WORKERS=max(1,min(8,os.cpu_count() or 1))

def ensure_parent_dir(path):
    directory=os.path.dirname(path)
    if directory:
        os.makedirs(directory,exist_ok=True)

def vec3_array_to_numpy(values):
    return np.array([[p[0],p[1],p[2]] for p in values],dtype=np.float32)

def read_usd_data(path):
    stage=Usd.Stage.Open(path)
    if stage is None:
        raise RuntimeError("failed to open USD:"+path)

    instancer=UsdGeom.PointInstancer(stage.GetPrimAtPath("/root/particles"))
    pos_attr=instancer.GetPositionsAttr()
    times=pos_attr.GetTimeSamples()
    if not times:
        raise RuntimeError("USD has no particle time samples:"+path)

    frames=[]
    for time_code in times:
        frames.append(vec3_array_to_numpy(pos_attr.Get(time_code)))

    curve=UsdGeom.BasisCurves(stage.GetPrimAtPath("/root/ground_curve"))
    ground=vec3_array_to_numpy(curve.GetPointsAttr().Get())
    return frames,ground

def frame_limits(frames,ground):
    all_points=np.concatenate(frames,axis=0)
    xmin=float(all_points[:,0].min())-0.08
    xmax=float(all_points[:,0].max())+0.08
    ground_y=float(ground[0,1])
    ymin=min(float(all_points[:,1].min()),ground_y)-0.06
    ymax=float(all_points[:,1].max())+0.06
    return (xmin,xmax),(ymin,ymax)

def render_frame(points,ground,xlim,ylim):
    ground_y=float(ground[0,1])
    fig=plt.figure(figsize=(6,6),dpi=120)
    ax=fig.add_subplot(111)
    ax.scatter(points[:,0],points[:,1],s=12,c="#66CCFF",edgecolors="none")
    ax.plot([xlim[0],xlim[1]],[ground_y,ground_y],linewidth=2.0,color="#FFFFFF")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal",adjustable="box")
    ax.set_facecolor("#112F41")
    fig.patch.set_facecolor("#112F41")
    ax.tick_params(colors="#D6E6EF")
    for spine in ax.spines.values():
        spine.set_color("#D6E6EF")
    fig.canvas.draw()
    image=np.asarray(fig.canvas.buffer_rgba())[...,:3].copy()
    plt.close(fig)
    return image

def render_frame_file(index,points,ground,xlim,ylim,path):
    image=render_frame(points,ground,xlim,ylim)
    imageio.imwrite(path,image)
    return index

def render_frames(frames,ground,xlim,ylim,directory,workers):
    paths=[os.path.join(directory,f"frame_{i:06d}.png") for i in range(len(frames))]
    workers=max(1,int(workers))
    if workers==1:
        for i,path in enumerate(paths):
            render_frame_file(i,frames[i],ground,xlim,ylim,path)
        return paths

    batch_size=workers*2
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for start in range(0,len(frames),batch_size):
            end=min(start+batch_size,len(frames))
            futures=[]
            for i in range(start,end):
                futures.append(executor.submit(render_frame_file,i,frames[i],ground,xlim,ylim,paths[i]))
            for future in futures:
                future.result()
    return paths

def save_video(usd_path=generate.DEFAULT_USD_PATH,
               video_path=DEFAULT_VIDEO_PATH,
               fps=generate.DEFAULT_FPS,
               workers=DEFAULT_WORKERS):
    frames,ground=read_usd_data(usd_path)
    xlim,ylim=frame_limits(frames,ground)
    ensure_parent_dir(video_path)
    with tempfile.TemporaryDirectory(prefix="basic_mpm_frames_") as directory:
        paths=render_frames(frames,ground,xlim,ylim,directory,workers)
        with imageio.get_writer(video_path,fps=fps,quality=8) as writer:
            for path in paths:
                writer.append_data(imageio.imread(path))
    return video_path

def main():
    usd_path=generate.save_particles_ground_usd(
        path=generate.DEFAULT_USD_PATH,
        fps=60,
        num_frames=2000,
        substeps_per_frame=10,
    )
    save_video(usd_path=usd_path,video_path=DEFAULT_VIDEO_PATH,fps=60)

if __name__=="__main__":
    main()
