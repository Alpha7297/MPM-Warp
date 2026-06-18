import os

os.makedirs("/tmp/mpmwarp_matplotlib_cache",exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR","/tmp/mpmwarp_matplotlib_cache")

import imageio.v2 as imageio
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pxr import Usd,UsdGeom

try:
    from . import generate
except ImportError:
    import generate

DEFAULT_VIDEO_DIR="outputs/videos"
CUBE_VIDEO_PATH=os.path.join(DEFAULT_VIDEO_DIR,"cube_compare.mp4")
TABLE_VIDEO_PATH=os.path.join(DEFAULT_VIDEO_DIR,"table_compare.mp4")

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

def frame_limits(frames_a,frames_b,ground_a,ground_b):
    all_points=np.concatenate(frames_a+frames_b,axis=0)
    xmin=float(all_points[:,0].min())-0.08
    xmax=float(all_points[:,0].max())+0.08
    ground_y=min(float(ground_a[0,1]),float(ground_b[0,1]))
    ymin=min(float(all_points[:,1].min()),ground_y)-0.06
    ymax=float(all_points[:,1].max())+0.06
    return (xmin,xmax),(ymin,ymax)

def draw_panel(ax,points,ground,xlim,ylim,title):
    ground_y=float(ground[0,1])
    ax.scatter(points[:,0],points[:,1],s=12,c="#66CCFF",edgecolors="none")
    ax.plot([xlim[0],xlim[1]],[ground_y,ground_y],linewidth=2.0,color="#FFFFFF")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal",adjustable="box")
    ax.set_title(title,color="#D6E6EF",fontsize=12)
    ax.set_facecolor("#112F41")
    ax.tick_params(colors="#D6E6EF")
    for spine in ax.spines.values():
        spine.set_color("#D6E6EF")

def render_compare_frame(points_a,points_b,ground_a,ground_b,xlim,ylim):
    fig=plt.figure(figsize=(12,6),dpi=120)
    fig.patch.set_facecolor("#112F41")
    ax_a=fig.add_subplot(121)
    ax_b=fig.add_subplot(122)
    draw_panel(ax_a,points_a,ground_a,xlim,ylim,"tradition")
    draw_panel(ax_b,points_b,ground_b,xlim,ylim,"nclaw")
    fig.tight_layout(pad=2.0)
    fig.canvas.draw()
    image=np.asarray(fig.canvas.buffer_rgba())[...,:3].copy()
    plt.close(fig)
    return image

def save_compare_video(tradition_path,nclaw_path,video_path,fps=generate.DEFAULT_FPS):
    frames_a,ground_a=read_usd_data(tradition_path)
    frames_b,ground_b=read_usd_data(nclaw_path)
    frame_count=min(len(frames_a),len(frames_b))
    xlim,ylim=frame_limits(frames_a[:frame_count],frames_b[:frame_count],ground_a,ground_b)
    ensure_parent_dir(video_path)
    with imageio.get_writer(video_path,fps=fps,quality=8) as writer:
        for i in range(frame_count):
            writer.append_data(render_compare_frame(frames_a[i],frames_b[i],ground_a,ground_b,xlim,ylim))
    return video_path

def plot_cube(tradition_path=generate.CUBE_TRADITION_USD,
              nclaw_path=generate.CUBE_NCLAW_USD,
              video_path=CUBE_VIDEO_PATH,
              fps=generate.DEFAULT_FPS):
    return save_compare_video(tradition_path,nclaw_path,video_path,fps)

def plot_table(tradition_path=generate.TABLE_TRADITION_USD,
               nclaw_path=generate.TABLE_NCLAW_USD,
               video_path=TABLE_VIDEO_PATH,
               fps=generate.DEFAULT_FPS):
    return save_compare_video(tradition_path,nclaw_path,video_path,fps)

def ensure_usd_inputs():
    paths=[
        generate.CUBE_TRADITION_USD,
        generate.CUBE_NCLAW_USD,
        generate.TABLE_TRADITION_USD,
        generate.TABLE_NCLAW_USD,
    ]
    missing=[path for path in paths if not os.path.exists(path)]
    if missing:
        print("missing USD files; generating:")
        for path in missing:
            print(path)
        generate.main()

def main():
    ensure_usd_inputs()
    cube_video=plot_cube()
    table_video=plot_table()
    print(cube_video)
    print(table_video)
    return cube_video,table_video

if __name__=="__main__":
    main()
