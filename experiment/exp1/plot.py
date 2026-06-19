import argparse
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor

os.makedirs("/tmp/mpmwarp_matplotlib_cache",exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR","/tmp/mpmwarp_matplotlib_cache")

import imageio.v2 as imageio
import matplotlib
import numpy as np
import warp as wp

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP1_DIR=os.path.dirname(__file__)
if EXP1_DIR not in sys.path:
    sys.path.insert(0,EXP1_DIR)

import train

DEFAULT_VIDEO_DIR="outputs/videos"
DEFAULT_WORKERS=max(1,min(8,os.cpu_count() or 1))
VIDEO_PATHS={
    1:os.path.join(DEFAULT_VIDEO_DIR,"exp11.mp4"),
    2:os.path.join(DEFAULT_VIDEO_DIR,"exp12.mp4"),
    3:os.path.join(DEFAULT_VIDEO_DIR,"exp13.mp4"),
}

def ensure_parent_dir(path):
    directory=os.path.dirname(path)
    if directory:
        os.makedirs(directory,exist_ok=True)

def video_path(model_id):
    return VIDEO_PATHS[int(model_id)]

def collect_frames(model,initial_velocity_values,gravity_y,device_name=None):
    if device_name is None:
        device_name=train.g.default_device_name()
    state=train.g.SimState(model,train.NUM_STEPS,device_name,requires_grad=False)
    initial_velocity=train.g.initial_velocity_array(model,device_name,initial_velocity_values)
    train.g.init_state(state,initial_velocity)
    train.rollout(state,train.NUM_STEPS,train.DT,gravity_y)
    wp.synchronize()
    return state.particle_pos.numpy().reshape(train.NUM_STEPS+1,model.num_particles,2).copy()

def frame_limits(frames,target):
    all_points=np.concatenate([frames.reshape(-1,2),target],axis=0)
    xmin=float(all_points[:,0].min())-0.08
    xmax=float(all_points[:,0].max())+0.08
    ymin=float(all_points[:,1].min())-0.08
    ymax=float(all_points[:,1].max())+0.08
    return (xmin,xmax),(ymin,ymax)

def render_frame(points,target,xlim,ylim):
    fig=plt.figure(figsize=(6,6),dpi=120)
    ax=fig.add_subplot(111)
    ax.scatter(points[:,0],points[:,1],s=12,c="#1D7AF3",alpha=1.0,edgecolors="none",zorder=2)
    ax.scatter(target[:,0],target[:,1],s=16,c="#FF2D2D",alpha=0.38,edgecolors="none",zorder=3)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal",adjustable="box")
    ax.set_facecolor("#0E1726")
    fig.patch.set_facecolor("#0E1726")
    ax.tick_params(colors="#D6E6EF")
    for spine in ax.spines.values():
        spine.set_color("#D6E6EF")
    fig.canvas.draw()
    image=np.asarray(fig.canvas.buffer_rgba())[...,:3].copy()
    plt.close(fig)
    return image

def render_frame_file(index,points,target,xlim,ylim,path):
    image=render_frame(points,target,xlim,ylim)
    imageio.imwrite(path,image)
    return index

def render_frames(frames,target,xlim,ylim,directory,workers):
    paths=[os.path.join(directory,f"frame_{i:06d}.png") for i in range(len(frames))]
    workers=max(1,int(workers))
    if workers==1:
        for i,path in enumerate(paths):
            render_frame_file(i,frames[i],target,xlim,ylim,path)
        return paths

    batch_size=workers*2
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for start in range(0,len(frames),batch_size):
            end=min(start+batch_size,len(frames))
            futures=[]
            for i in range(start,end):
                futures.append(executor.submit(render_frame_file,i,frames[i],target,xlim,ylim,paths[i]))
            for future in futures:
                future.result()
    return paths

def save_video(model_id=1,output_path=None,device_name=None,workers=DEFAULT_WORKERS):
    model_id=int(model_id)
    model,_,_,gravity_y=train.case_initial_conditions(model_id)
    checkpoint=train.latest_checkpoint_path(model_id)
    if checkpoint is None:
        print(f"checkpoint not found for model {model_id} in {train.NET_DIR}")
        sys.exit(1)
    velocity,epoch,loss=train.load_checkpoint(checkpoint)
    with np.load(checkpoint) as data:
        target=data["target"].astype(np.float32).copy()
    if velocity.shape!=(model.num_particles,2):
        raise ValueError("checkpoint velocity shape does not match model")
    frames=collect_frames(model,velocity,gravity_y,device_name)
    xlim,ylim=frame_limits(frames,target)
    if output_path is None:
        output_path=video_path(model_id)
    ensure_parent_dir(output_path)
    with tempfile.TemporaryDirectory(prefix="exp1_frames_") as directory:
        paths=render_frames(frames,target,xlim,ylim,directory,workers)
        with imageio.get_writer(output_path,fps=train.g.DEFAULT_FPS,quality=8) as writer:
            for path in paths:
                writer.append_data(imageio.imread(path))
    print(f"loaded={checkpoint} epoch={epoch} loss={loss:.6e}")
    print(output_path)
    return output_path

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",type=int,default=1,choices=[1,2,3])
    return parser.parse_args()

def main():
    args=parse_args()
    return save_video(args.model)

if __name__=="__main__":
    main()
