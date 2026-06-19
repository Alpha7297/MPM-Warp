import argparse
import gc
import os
import tempfile

os.makedirs("/tmp/mpmwarp_matplotlib_cache",exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR","/tmp/mpmwarp_matplotlib_cache")

import imageio.v2 as imageio
import matplotlib
import numpy as np

if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY") and os.name!="nt":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

import generate

DEFAULT_RENDER="show"
DEFAULT_VIDEO_TEMPLATE="outputs/3d_mpm/{model}_video.mp4"
DEFAULT_VIDEO_PATH=DEFAULT_VIDEO_TEMPLATE.format(model=generate.DEFAULT_MODEL)
TMP_FRAME_DIR="/tmp"
TABLE_COLOR="#65C7F7"
CUBE_COLOR="#F2A65A"
GROUND_COLOR="#D6E6EF"
BACKGROUND_COLOR="#101820"

def default_video_path(model=generate.DEFAULT_MODEL):
    if not isinstance(model,str):
        model=model.name
    return DEFAULT_VIDEO_TEMPLATE.format(model=model)

def ensure_parent_dir(path):
    directory=os.path.dirname(path)
    if directory:
        os.makedirs(directory,exist_ok=True)

def load_frames(path=None,
                model=None,
                num_frames=generate.DEFAULT_NUM_FRAMES,
                substeps_per_frame=generate.DEFAULT_SUBSTEPS_PER_FRAME,
                frame_dt=generate.DEFAULT_FRAME_DT,
                sample_count=generate.DEFAULT_SAMPLE_COUNT,
                sample_seed=0,
                fps=generate.DEFAULT_FPS,
                device_name=generate.k.device):
    model=generate.resolve_model(model)
    if path is None:
        path=generate.default_data_path(model)
    if not os.path.exists(path):
        generate.save_frames_npz(
            path=path,
            model=model,
            fps=fps,
            num_frames=num_frames,
            substeps_per_frame=substeps_per_frame,
            frame_dt=frame_dt,
            sample_count=sample_count,
            sample_seed=sample_seed,
            device_name=device_name,
        )
    data=np.load(path)
    frames=data["frames"]
    labels=data["labels"]
    ground=float(data["ground"])
    grid_shape=data["grid_shape"].astype(np.int32)
    grid_size=float(data["grid_size"])
    stored_fps=int(data["fps"])
    return frames,labels,ground,grid_shape,grid_size,stored_fps

def axis_limits(grid_shape,grid_size,ground):
    xmax=float(grid_shape[0]-1)*grid_size
    ymax=float(grid_shape[1]-1)*grid_size
    zmax=float(grid_shape[2]-1)*grid_size
    return (0.0,xmax),(ground,ymax),(0.0,zmax)

def setup_axes(fig,grid_shape,grid_size,ground):
    ax=fig.add_subplot(111,projection="3d")
    xlim,vertical_lim,zlim=axis_limits(grid_shape,grid_size,ground)
    ax.set_xlim(xlim)
    ax.set_ylim(zlim)
    ax.set_zlim(vertical_lim)
    ax.set_box_aspect((xlim[1]-xlim[0],zlim[1]-zlim[0],vertical_lim[1]-vertical_lim[0]))
    ax.set_facecolor(BACKGROUND_COLOR)
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    ax.tick_params(colors=GROUND_COLOR)
    ax.xaxis.label.set_color(GROUND_COLOR)
    ax.yaxis.label.set_color(GROUND_COLOR)
    ax.zaxis.label.set_color(GROUND_COLOR)
    ax.xaxis.pane.set_facecolor((0.06,0.09,0.12,1.0))
    ax.yaxis.pane.set_facecolor((0.06,0.09,0.12,1.0))
    ax.zaxis.pane.set_facecolor((0.06,0.09,0.12,1.0))
    ax.view_init(elev=22,azim=-55)
    ground_points=generate.ground_square()
    ax.plot(ground_points[:,0],ground_points[:,2],ground_points[:,1],color=GROUND_COLOR,linewidth=1.5)
    return ax

def colors_for_labels(labels):
    colors=np.empty(labels.shape[0],dtype=object)
    colors[labels==0]=TABLE_COLOR
    colors[labels!=0]=CUBE_COLOR
    return colors

def make_animation(frames,labels,ground,grid_shape,grid_size,interval=33,point_size=2.0):
    colors=colors_for_labels(labels)
    fig=plt.figure(figsize=(7,7),dpi=110)
    ax=setup_axes(fig,grid_shape,grid_size,ground)
    first=frames[0]
    scatter=ax.scatter(first[:,0],first[:,2],first[:,1],s=point_size,c=colors,edgecolors="none",depthshade=False)

    def update(frame_i):
        points=frames[frame_i]
        scatter._offsets3d=(points[:,0],points[:,2],points[:,1])
        return (scatter,)

    return animation.FuncAnimation(fig,update,frames=len(frames),interval=interval,blit=False)

def show_animation(data_path=None,
                   model=None,
                   fps=generate.DEFAULT_FPS,
                   num_frames=generate.DEFAULT_NUM_FRAMES,
                   substeps_per_frame=generate.DEFAULT_SUBSTEPS_PER_FRAME,
                   frame_dt=generate.DEFAULT_FRAME_DT,
                   sample_count=generate.DEFAULT_SAMPLE_COUNT,
                   sample_seed=0,
                   device_name=generate.k.device):
    model=generate.resolve_model(model)
    if data_path is None:
        data_path=generate.default_data_path(model)
    if not os.path.exists(data_path):
        generate.save_frames_npz(
            path=data_path,
            model=model,
            fps=fps,
            num_frames=num_frames,
            substeps_per_frame=substeps_per_frame,
            frame_dt=frame_dt,
            sample_count=sample_count,
            sample_seed=sample_seed,
            device_name=device_name,
        )
    frames,labels,ground,grid_shape,grid_size,stored_fps=load_frames(
        path=data_path,
        model=model,
        num_frames=num_frames,
        substeps_per_frame=substeps_per_frame,
        frame_dt=frame_dt,
        sample_count=sample_count,
        sample_seed=sample_seed,
        fps=fps,
        device_name=device_name,
    )
    interval=1000.0/float(stored_fps)
    anim=make_animation(frames,labels,ground,grid_shape,grid_size,interval=interval)
    plt.show()
    return anim

def render_frame(points,labels,ground,grid_shape,grid_size,point_size=2.0):
    colors=colors_for_labels(labels)
    fig=plt.figure(figsize=(7,7),dpi=128)
    ax=setup_axes(fig,grid_shape,grid_size,ground)
    ax.scatter(points[:,0],points[:,2],points[:,1],s=point_size,c=colors,edgecolors="none",depthshade=False)
    fig.canvas.draw()
    image=np.asarray(fig.canvas.buffer_rgba())[...,:3].copy()
    plt.close(fig)
    return image

def save_frame_image(path,points,labels,ground,grid_shape,grid_size,point_size=2.0):
    colors=colors_for_labels(labels)
    fig=plt.figure(figsize=(7,7),dpi=128)
    try:
        ax=setup_axes(fig,grid_shape,grid_size,ground)
        ax.scatter(points[:,0],points[:,2],points[:,1],s=point_size,c=colors,edgecolors="none",depthshade=False)
        fig.savefig(path,format="png",facecolor=fig.get_facecolor())
    finally:
        plt.close(fig)
        gc.collect()

def save_video(data_path=None,
               video_path=None,
               model=None,
               fps=generate.DEFAULT_FPS,
               num_frames=generate.DEFAULT_NUM_FRAMES,
               substeps_per_frame=generate.DEFAULT_SUBSTEPS_PER_FRAME,
               frame_dt=generate.DEFAULT_FRAME_DT,
               sample_count=generate.DEFAULT_SAMPLE_COUNT,
               sample_seed=0,
               device_name=generate.k.device):
    model=generate.resolve_model(model)
    if data_path is None:
        data_path=generate.default_data_path(model)
    if video_path is None:
        video_path=default_video_path(model)
    if not os.path.exists(data_path):
        generate.save_frames_npz(
            path=data_path,
            model=model,
            fps=fps,
            num_frames=num_frames,
            substeps_per_frame=substeps_per_frame,
            frame_dt=frame_dt,
            sample_count=sample_count,
            sample_seed=sample_seed,
            device_name=device_name,
        )
    frames,labels,ground,grid_shape,grid_size,stored_fps=load_frames(
        path=data_path,
        model=model,
        num_frames=num_frames,
        substeps_per_frame=substeps_per_frame,
        frame_dt=frame_dt,
        sample_count=sample_count,
        sample_seed=sample_seed,
        fps=fps,
        device_name=device_name,
    )
    ensure_parent_dir(video_path)
    with tempfile.TemporaryDirectory(prefix="mpm3d_frames_",dir=TMP_FRAME_DIR) as directory:
        frame_path=os.path.join(directory,"frame.png")
        with imageio.get_writer(video_path,fps=stored_fps,quality=8) as writer:
            for points in frames:
                save_frame_image(frame_path,points,labels,ground,grid_shape,grid_size)
                image=imageio.imread(frame_path)
                writer.append_data(image)
                del image
                os.remove(frame_path)
                gc.collect()
    return video_path

def build_parser():
    parser=argparse.ArgumentParser(description="Render sampled 3D MPM frames.")
    parser.add_argument("--model",choices=["cube","table"],default=generate.DEFAULT_MODEL)
    parser.add_argument("--render",choices=["show","video"],default=DEFAULT_RENDER)
    parser.add_argument("--data-path",default=None)
    parser.add_argument("--video-path",default=None)
    parser.add_argument("--fps",type=int,default=generate.DEFAULT_FPS)
    parser.add_argument("--num-frames",type=int,default=generate.DEFAULT_NUM_FRAMES)
    parser.add_argument("--substeps-per-frame",type=int,default=generate.DEFAULT_SUBSTEPS_PER_FRAME)
    parser.add_argument("--frame-dt",type=float,default=generate.DEFAULT_FRAME_DT)
    parser.add_argument("--sample-count",type=int,default=generate.DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--sample-seed",type=int,default=0)
    parser.add_argument("--device",default=generate.k.device)
    return parser

def main(argv=None):
    args=build_parser().parse_args(argv)
    model=generate.model_from_name(args.model)
    data_path=args.data_path
    if data_path is None:
        data_path=generate.default_data_path(args.model)
    if args.render=="show":
        return show_animation(
            data_path=data_path,
            model=model,
            fps=args.fps,
            num_frames=args.num_frames,
            substeps_per_frame=args.substeps_per_frame,
            frame_dt=args.frame_dt,
            sample_count=args.sample_count,
            sample_seed=args.sample_seed,
            device_name=args.device,
        )
    video_path=args.video_path
    if video_path is None:
        video_path=default_video_path(args.model)
    video_path=save_video(
        data_path=data_path,
        video_path=video_path,
        model=model,
        fps=args.fps,
        num_frames=args.num_frames,
        substeps_per_frame=args.substeps_per_frame,
        frame_dt=args.frame_dt,
        sample_count=args.sample_count,
        sample_seed=args.sample_seed,
        device_name=args.device,
    )
    print(video_path)
    return video_path

if __name__=="__main__":
    main()
