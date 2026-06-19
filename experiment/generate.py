import os
from pathlib import Path

import numpy as np
import warp as wp
import warp.render as render
from pxr import Gf,UsdGeom

import kernels as k

PARTICLE_SPACING=0.01
DENSITY=1000.0
V0=PARTICLE_SPACING*PARTICLE_SPACING
M0=DENSITY*V0
GROUND=0.299
RESTITUTION=0.7
DAMPING=0.9
BASE_DIR=Path(__file__).resolve().parent
DEFAULT_FPS=60
DEFAULT_NUM_STEPS=2000
DEFAULT_DT=5.0e-4
DEFAULT_OUTPUT_DIR="outputs/nclaw"
NET_PATH=BASE_DIR/"net/3000.npz"
CUBE_TRADITION_USD=os.path.join(DEFAULT_OUTPUT_DIR,"cube_tradition.usd")
CUBE_NCLAW_USD=os.path.join(DEFAULT_OUTPUT_DIR,"cube_nclaw.usd")
TABLE_TRADITION_USD=os.path.join(DEFAULT_OUTPUT_DIR,"table_tradition.usd")
TABLE_NCLAW_USD=os.path.join(DEFAULT_OUTPUT_DIR,"table_nclaw.usd")

RECT_COL=25
RECT_ROW=20
RECT_ORIGIN=(0.88,1.05)

TABLE_TOP_COL=30
TABLE_TOP_ROW=10
TABLE_LEG_COL=10
TABLE_LEG_ROW=10
TABLE_ORIGIN=(0.855,1.105)
TABLE_ROTATION=0.1*np.pi

class ModelSpec:
    def __init__(self,name,positions,particle_spacing=PARTICLE_SPACING,density=DENSITY):
        self.name=name
        self.positions=positions.astype(np.float32)
        self.particle_spacing=particle_spacing
        self.density=density
        self.v0=particle_spacing*particle_spacing
        self.m0=density*self.v0
        self.num_particles=self.positions.shape[0]

def rectangle_positions(origin=RECT_ORIGIN,
                        col=RECT_COL,
                        row=RECT_ROW,
                        particle_spacing=PARTICLE_SPACING):
    positions=np.zeros((col*row,2),dtype=np.float32)
    for i in range(col*row):
        x=i%col
        y=i//col
        positions[i,0]=origin[0]+float(x)*particle_spacing
        positions[i,1]=origin[1]+float(y)*particle_spacing
    return positions

def table_positions(origin=TABLE_ORIGIN,
                    particle_spacing=PARTICLE_SPACING,
                    table_top_col=TABLE_TOP_COL,
                    table_top_row=TABLE_TOP_ROW,
                    table_leg_col=TABLE_LEG_COL,
                    table_leg_row=TABLE_LEG_ROW,
                    rotation=TABLE_ROTATION):
    num_particles=table_top_col*table_top_row+2*table_leg_col*table_leg_row
    positions=np.zeros((num_particles,2),dtype=np.float32)
    center=np.array([
        origin[0]+float(table_top_col-1)*(0.5*particle_spacing),
        origin[1]+float(table_top_row+table_leg_row-1)*(0.5*particle_spacing),
    ],dtype=np.float32)
    c=np.cos(rotation)
    s=np.sin(rotation)
    for i in range(num_particles):
        if i<table_top_col*table_top_row:
            x=i%table_top_col
            y=i//table_top_col+table_leg_row
        else:
            leg_i=i-table_top_col*table_top_row
            if leg_i<table_leg_col*table_leg_row:
                x=leg_i%table_leg_col
                y=leg_i//table_leg_col
            else:
                leg_i=leg_i-table_leg_col*table_leg_row
                x=leg_i%table_leg_col+table_top_col-table_leg_col
                y=leg_i//table_leg_col
        pos=np.array([
            origin[0]+float(x)*particle_spacing,
            origin[1]+float(y)*particle_spacing,
        ],dtype=np.float32)
        local=pos-center
        positions[i,0]=center[0]+c*local[0]-s*local[1]
        positions[i,1]=center[1]+s*local[0]+c*local[1]
    return positions

def rectangle_model():
    return ModelSpec("rectangle",rectangle_positions())

def table_model():
    return ModelSpec("table",table_positions())

class SimState:
    def __init__(self,model,total_steps,device_name=k.device,requires_grad=False):
        self.model=model
        self.device=device_name
        self.total_steps=total_steps
        self.num_particles=model.num_particles
        particle_count=(total_steps+1)*model.num_particles
        grid_count=total_steps*int(k.NUM_GRIDS)
        self.particle_pos=wp.zeros(particle_count,dtype=wp.vec2,device=device_name,requires_grad=requires_grad)
        self.particle_vel=wp.zeros(particle_count,dtype=wp.vec2,device=device_name,requires_grad=requires_grad)
        self.particle_vel_trial=wp.zeros(total_steps*model.num_particles,dtype=wp.vec2,device=device_name,requires_grad=requires_grad)
        self.particle_dvel=wp.zeros(particle_count,dtype=wp.mat22,device=device_name,requires_grad=requires_grad)
        self.particle_F=wp.zeros(particle_count,dtype=wp.mat22,device=device_name,requires_grad=requires_grad)
        self.particle_P=wp.zeros(particle_count,dtype=wp.mat22,device=device_name,requires_grad=requires_grad)
        self.grid_momentum=wp.zeros(grid_count,dtype=wp.vec2,device=device_name,requires_grad=requires_grad)
        self.grid_f=wp.zeros(grid_count,dtype=wp.vec2,device=device_name,requires_grad=requires_grad)
        self.grid_mass=wp.zeros(grid_count,dtype=float,device=device_name,requires_grad=requires_grad)
        self.grid_vel=wp.zeros(grid_count,dtype=wp.vec2,device=device_name,requires_grad=requires_grad)

    def zero_forward(self):
        self.particle_pos.zero_()
        self.particle_vel.zero_()
        self.particle_vel_trial.zero_()
        self.particle_dvel.zero_()
        self.particle_F.zero_()
        self.particle_P.zero_()
        self.grid_momentum.zero_()
        self.grid_f.zero_()
        self.grid_mass.zero_()
        self.grid_vel.zero_()

    def zero_grid(self):
        self.grid_momentum.zero_()
        self.grid_f.zero_()
        self.grid_mass.zero_()
        self.grid_vel.zero_()

    def zero_grad(self):
        for array in self.arrays():
            if array.grad is not None:
                array.grad.zero_()

    def arrays(self):
        return [
            self.particle_pos,
            self.particle_vel,
            self.particle_vel_trial,
            self.particle_dvel,
            self.particle_F,
            self.particle_P,
            self.grid_momentum,
            self.grid_f,
            self.grid_mass,
            self.grid_vel,
        ]

def default_device_name():
    if wp.is_cuda_available():
        return "cuda:0"
    return "cpu"

def initial_velocity_array(model,device_name,values=None):
    if values is None:
        values=np.zeros((model.num_particles,2),dtype=np.float32)
    return wp.array(values,dtype=wp.vec2,device=device_name)

def init_state(state,initial_velocity=None):
    if initial_velocity is None:
        initial_velocity=initial_velocity_array(state.model,state.device)
    initial_pos=wp.array(state.model.positions,dtype=wp.vec2,device=state.device)
    wp.launch(
        k.init_particle_state,
        dim=state.num_particles,
        inputs=[
            initial_pos,
            initial_velocity,
            state.particle_pos,
            state.particle_vel,
            state.particle_dvel,
            state.particle_F,
            state.particle_P,
            0,
            state.num_particles,
        ],
        device=state.device,
    )

def ensure_parent_dir(path):
    directory=os.path.dirname(path)
    if directory:
        os.makedirs(directory,exist_ok=True)

def particle_points(state,t):
    offset=t*state.num_particles
    xy=state.particle_pos.numpy()[offset:offset+state.num_particles]
    points=np.zeros((xy.shape[0],3),dtype=np.float32)
    points[:,0]=xy[:,0]
    points[:,1]=xy[:,1]
    return points

def ground_points():
    return np.array([
        [0.0,GROUND,0.0],
        [float(k.GRID_LEN)*float(k.GRID_SIZE),GROUND,0.0],
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

def render_state(renderer,state,t,frame,fps):
    renderer.begin_frame(frame/float(fps))
    renderer.render_points(
        "particles",
        points=particle_points(state,t),
        radius=0.003,
        colors=(0.4,0.8,1.0),
        as_spheres=True,
    )
    renderer.end_frame()

def substep(state,dt,mode,t,net=None):
    if mode=="tradition":
        wp.launch(
            k.theory_strain,
            dim=state.num_particles,
            inputs=[state.particle_dvel,state.particle_F,state.particle_P,t,state.num_particles,dt,k.MU,k.LAMBDA],
            device=state.device,
        )
    elif mode=="nclaw":
        wp.launch(
            k.update_F,
            dim=state.num_particles,
            inputs=[state.particle_dvel,state.particle_F,t,state.num_particles,dt],
            device=state.device,
        )
        net.forward(state.particle_F,state.particle_P,t)
    else:
        raise ValueError("unknown mode:"+mode)

    wp.launch(
        k.P2G_update_grid,
        dim=state.num_particles,
        inputs=[
            state.grid_momentum,
            state.grid_f,
            state.grid_mass,
            state.particle_pos,
            state.particle_vel,
            state.particle_F,
            state.particle_P,
            t,
            state.num_particles,
            int(k.NUM_GRIDS),
            state.model.m0,
            k.GRID_SIZE,
            k.GRID_LEN,
            k.GRID_HEI,
            state.model.v0,
        ],
        device=state.device,
    )
    wp.launch(
        k.update_grid_vel,
        dim=int(k.NUM_GRIDS),
        inputs=[state.grid_momentum,state.grid_f,state.grid_mass,state.grid_vel,t,int(k.NUM_GRIDS),wp.vec2(0.0,-k.GRAVITY),dt],
        device=state.device,
    )
    wp.launch(
        k.G2P,
        dim=state.num_particles,
        inputs=[state.grid_vel,state.particle_pos,state.particle_vel_trial,state.particle_dvel,t,state.num_particles,int(k.NUM_GRIDS),k.GRID_SIZE,k.GRID_LEN,k.GRID_HEI],
        device=state.device,
    )
    wp.launch(
        k.update_pos,
        dim=state.num_particles,
        inputs=[state.particle_pos,state.particle_vel,state.particle_vel_trial,t,state.num_particles,dt,GROUND,RESTITUTION,DAMPING],
        device=state.device,
    )

def roll_stream_state(state):
    wp.launch(
        k.copy_particle_state,
        dim=state.num_particles,
        inputs=[state.particle_pos,state.particle_vel,state.particle_dvel,state.particle_F,state.particle_P,0,state.num_particles],
        device=state.device,
    )
    state.zero_grid()

def save_usd(model,path,mode,num_steps=DEFAULT_NUM_STEPS,dt=DEFAULT_DT,fps=DEFAULT_FPS,device_name=None,net_path=None):
    ensure_parent_dir(path)
    if device_name is None:
        device_name=default_device_name()
    state=SimState(model,1,device_name)
    init_state(state)
    net=None
    renderer=render.UsdRenderer(path,up_axis="Y",fps=fps,scaling=1.0)
    ground=ground_points()
    renderer.render_line_strip("ground",vertices=ground,color=(1.0,1.0,1.0),radius=0.004)
    write_ground_curve(renderer.stage,ground)
    render_state(renderer,state,0,0,fps)
    for step in range(1,num_steps+1):
        substep(state,dt,mode,0,net)
        wp.synchronize()
        render_state(renderer,state,1,step,fps)
        roll_stream_state(state)
    renderer.save()
    return path

def generate_cube(output_dir=DEFAULT_OUTPUT_DIR,num_steps=DEFAULT_NUM_STEPS,dt=DEFAULT_DT,fps=DEFAULT_FPS,device_name=None,net_path=None):
    model=rectangle_model()
    tradition_path=os.path.join(output_dir,"cube_tradition.usd")
    nclaw_path=os.path.join(output_dir,"cube_nclaw.usd")
    save_usd(model,tradition_path,"tradition",num_steps,dt,fps,device_name,net_path)
    save_usd(model,nclaw_path,"nclaw",num_steps,dt,fps,device_name,net_path)
    return tradition_path,nclaw_path

def generate_table(output_dir=DEFAULT_OUTPUT_DIR,num_steps=DEFAULT_NUM_STEPS,dt=DEFAULT_DT,fps=DEFAULT_FPS,device_name=None,net_path=None):
    model=table_model()
    tradition_path=os.path.join(output_dir,"table_tradition.usd")
    nclaw_path=os.path.join(output_dir,"table_nclaw.usd")
    save_usd(model,tradition_path,"tradition",num_steps,dt,fps,device_name,net_path)
    save_usd(model,nclaw_path,"nclaw",num_steps,dt,fps,device_name,net_path)
    return tradition_path,nclaw_path

def main():
    cube_paths=generate_cube()
    table_paths=generate_table()
    print(cube_paths)
    print(table_paths)
    return cube_paths,table_paths

if __name__=="__main__":
    main()
