import os

import numpy as np
import warp as wp
import warp.render as render
from pxr import Gf,UsdGeom

import kernels as k

DEFAULT_FPS=60
DEFAULT_NUM_FRAMES=2000
DEFAULT_SUBSTEPS_PER_FRAME=10
DEFAULT_FRAME_DT=0.001
DEFAULT_USD_PATH="outputs/mpm.usd"

class SimState:
    def __init__(self,device_name=k.device):
        self.device=device_name
        self.grid_pos=wp.zeros(k.NUM_GRIDS,dtype=wp.vec2,device=device_name)
        self.grid_vel=wp.zeros(k.NUM_GRIDS,dtype=wp.vec2,device=device_name)
        self.grid_f=wp.zeros(k.NUM_GRIDS,dtype=wp.vec2,device=device_name)
        self.grid_mass=wp.zeros(k.NUM_GRIDS,dtype=float,device=device_name)
        self.particle_pos=wp.zeros(k.NUM_PARTICLES,dtype=wp.vec2,device=device_name)
        self.particle_vel=wp.zeros(k.NUM_PARTICLES,dtype=wp.vec2,device=device_name)
        self.particle_dvel=wp.zeros(k.NUM_PARTICLES,dtype=wp.mat22,device=device_name)
        self.particle_F=wp.zeros(k.NUM_PARTICLES,dtype=wp.mat22,device=device_name)
        self.particle_P=wp.zeros(k.NUM_PARTICLES,dtype=wp.mat22,device=device_name)
        self.particle_mass=wp.zeros(k.NUM_PARTICLES,dtype=float,device=device_name)
        self.grid_idx=wp.zeros(k.NUM_PARTICLES,dtype=int,device=device_name)

def ensure_parent_dir(path):
    directory=os.path.dirname(path)
    if directory:
        os.makedirs(directory,exist_ok=True)

def init_state(state):
    wp.launch(
        k.init_grid,
        dim=k.NUM_GRIDS,
        inputs=[state.grid_pos,state.grid_vel,state.grid_f,state.grid_mass,k.GRID_SIZE,k.GRID_LEN],
        device=state.device,
    )
    wp.launch(
        k.init_particle,
        dim=k.NUM_PARTICLES,
        inputs=[
            state.particle_pos,
            state.particle_vel,
            state.particle_dvel,
            state.particle_F,
            state.particle_P,
            state.particle_mass,
            state.grid_idx,
            wp.vec2(0.855,1.105),
            k.PARTICLE_SPACING,
            k.TABLE_TOP_COL,
            k.TABLE_TOP_ROW,
            k.TABLE_LEG_COL,
            k.TABLE_LEG_ROW,
            k.TABLE_TOP_PARTICLES,
            k.TABLE_LEG_PARTICLES,
            k.M0,
            k.GRID_SIZE,
            k.GRID_LEN,
        ],
        device=state.device,
    )

def substep(state,dt):
    wp.launch(
        k.zerolize_grids,
        dim=k.NUM_GRIDS,
        inputs=[state.grid_vel,state.grid_f,state.grid_mass],
        device=state.device,
    )
    wp.launch(
        k.P2G_update_grid,
        dim=k.NUM_PARTICLES,
        inputs=[
            state.grid_pos,
            state.grid_vel,
            state.grid_f,
            state.grid_mass,
            state.particle_pos,
            state.particle_vel,
            state.particle_mass,
            state.particle_F,
            state.particle_P,
            k.GRID_SIZE,
            k.GRID_LEN,
            k.GRID_HEI,
            k.V0,
        ],
        device=state.device,
    )
    wp.launch(
        k.P2G_grid_vel,
        dim=k.NUM_GRIDS,
        inputs=[state.grid_vel,state.grid_mass],
        device=state.device,
    )
    wp.launch(
        k.apply_particle_ground_contact,
        dim=k.NUM_PARTICLES,
        inputs=[
            state.grid_pos,
            state.grid_vel,
            state.grid_mass,
            state.particle_pos,
            state.particle_mass,
            k.GRID_SIZE,
            k.GRID_LEN,
            k.GRID_HEI,
            dt,
            k.GROUND,
            k.RESTITUTION,
            k.DAMPING,
        ],
        device=state.device,
    )
    wp.launch(
        k.update_grid_vel,
        dim=k.NUM_GRIDS,
        inputs=[state.grid_vel,state.grid_f,state.grid_mass,wp.vec2(0.0,-k.GRAVITY),dt],
        device=state.device,
    )
    wp.launch(
        k.G2P,
        dim=k.NUM_PARTICLES,
        inputs=[
            state.grid_pos,
            state.grid_vel,
            state.particle_pos,
            state.particle_vel,
            state.particle_dvel,
            k.GRID_SIZE,
            k.GRID_LEN,
            k.GRID_HEI,
        ],
        device=state.device,
    )
    wp.launch(
        k.update_pos,
        dim=k.NUM_PARTICLES,
        inputs=[
            state.particle_pos,
            state.particle_vel,
            state.particle_dvel,
            state.particle_F,
            state.particle_P,
            state.grid_idx,
            k.GRID_SIZE,
            k.GRID_LEN,
            dt,
            k.GROUND,
            k.MU,
            k.LAMBDA,
        ],
        device=state.device,
    )

def warmup(device_name=k.device):
    state=SimState(device_name)
    init_state(state)
    substep(state,DEFAULT_FRAME_DT/float(DEFAULT_SUBSTEPS_PER_FRAME))
    wp.synchronize()

def particle_points(state):
    xy=state.particle_pos.numpy()
    points=np.zeros((xy.shape[0],3),dtype=np.float32)
    points[:,0]=xy[:,0]
    points[:,1]=xy[:,1]
    return points

def ground_points():
    return np.array([
        [0.0,float(k.GROUND),0.0],
        [float(k.GRID_LEN)*float(k.GRID_SIZE),float(k.GROUND),0.0],
    ],dtype=np.float32)

def write_ground_curve(stage,points):
    curve=UsdGeom.BasisCurves.Define(stage,"/root/ground_curve")
    curve.CreateTypeAttr(UsdGeom.Tokens.linear)
    curve.CreateCurveVertexCountsAttr([2])
    curve.CreatePointsAttr([Gf.Vec3f(float(points[0,0]),float(points[0,1]),float(points[0,2])),
                            Gf.Vec3f(float(points[1,0]),float(points[1,1]),float(points[1,2]))])
    curve.CreateWidthsAttr([0.004])
    curve.SetWidthsInterpolation(UsdGeom.Tokens.constant)
    UsdGeom.Gprim(curve).CreateDisplayColorAttr([Gf.Vec3f(1.0,1.0,1.0)])

def save_particles_ground_usd(path=DEFAULT_USD_PATH,
                              fps=DEFAULT_FPS,
                              num_frames=DEFAULT_NUM_FRAMES,
                              substeps_per_frame=DEFAULT_SUBSTEPS_PER_FRAME,
                              frame_dt=DEFAULT_FRAME_DT,
                              device_name=k.device):
    ensure_parent_dir(path)
    warmup(device_name)

    state=SimState(device_name)
    init_state(state)

    renderer=render.UsdRenderer(path,up_axis="Y",fps=fps,scaling=1.0)
    ground=ground_points()
    renderer.render_line_strip("ground",vertices=ground,color=(1.0,1.0,1.0),radius=0.004)
    write_ground_curve(renderer.stage,ground)

    dt=frame_dt/float(substeps_per_frame)
    for frame in range(num_frames):
        for _ in range(substeps_per_frame):
            substep(state,dt)
        wp.synchronize()
        renderer.begin_frame(frame/float(fps))
        renderer.render_points(
            "particles",
            points=particle_points(state),
            radius=0.003,
            colors=(0.4,0.8,1.0),
            as_spheres=True,
        )
        renderer.end_frame()

    renderer.save()
    return path

def main():
    save_particles_ground_usd()

if __name__=="__main__":
    main()
