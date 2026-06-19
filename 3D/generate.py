import argparse
import os

import numpy as np
import warp as wp

import kernels as k

DEFAULT_FPS=60
DEFAULT_NUM_FRAMES=2000
DEFAULT_SUBSTEPS_PER_FRAME=10
DEFAULT_FRAME_DT=0.002
DEFAULT_SAMPLE_COUNT=40000
DEFAULT_MODEL="cube"
DEFAULT_NPZ_TEMPLATE="outputs/3d_mpm/{model}_frames.npz"
DEFAULT_NPZ_PATH=DEFAULT_NPZ_TEMPLATE.format(model=DEFAULT_MODEL)
GRID_BOUNDARY=3
INITIAL_ROTATION_Z_DEGREES=45.0

NCLAW_WORLD_SIZE=0.014*160.0
WORLD_SIZE=float(k.GRID_SIZE)*float(k.GRID_LEN)
SCENE_SCALE=WORLD_SIZE/NCLAW_WORLD_SIZE
GROUND_Y=float(k.GROUND)
PARTICLE_SPACING=float(k.PARTICLE_SPACING)
DENSITY=float(k.DENSITY)

TABLE_TOP_COL=28
TABLE_TOP_ROW=10
TABLE_TOP_DEP=28
TABLE_LEG_COL=10
TABLE_LEG_ROW=10
TABLE_LEG_DEP=10
TABLE_ORIGIN=(0.855*SCENE_SCALE,1.105*SCENE_SCALE,0.855*SCENE_SCALE)

CUBE_COL=24
CUBE_ROW=18
CUBE_DEP=24
CUBE_ORIGIN=(0.88*SCENE_SCALE,1.05*SCENE_SCALE,0.98*SCENE_SCALE)

TABLE_LABEL=0
CUBE_LABEL=1

class ModelSpec:
    def __init__(self,name,positions,labels,particle_spacing=PARTICLE_SPACING,density=DENSITY):
        self.name=name
        self.positions=positions.astype(np.float32)
        self.labels=labels.astype(np.int32)
        self.particle_spacing=particle_spacing
        self.density=density
        self.v0=particle_spacing*particle_spacing*particle_spacing
        self.m0=density*self.v0
        self.num_particles=self.positions.shape[0]

def box_positions(origin,col,row,dep,particle_spacing=PARTICLE_SPACING):
    positions=np.zeros((col*row*dep,3),dtype=np.float32)
    for i in range(col*row*dep):
        x=i%col
        y=(i//col)%row
        z=i//(col*row)
        positions[i,0]=origin[0]+float(x)*particle_spacing
        positions[i,1]=origin[1]+float(y)*particle_spacing
        positions[i,2]=origin[2]+float(z)*particle_spacing
    return positions

def rotate_positions_z_clockwise(positions,angle_degrees=INITIAL_ROTATION_Z_DEGREES):
    angle=np.deg2rad(float(angle_degrees))
    if angle==0.0:
        return positions
    center=positions.mean(axis=0)
    shifted=positions-center
    cos_angle=np.cos(angle)
    sin_angle=np.sin(angle)
    rotated=shifted.copy()
    rotated[:,0]=shifted[:,0]*cos_angle+shifted[:,1]*sin_angle
    rotated[:,1]=-shifted[:,0]*sin_angle+shifted[:,1]*cos_angle
    return rotated+center

def cube_positions(origin=CUBE_ORIGIN,
                   col=CUBE_COL,
                   row=CUBE_ROW,
                   dep=CUBE_DEP,
                   particle_spacing=PARTICLE_SPACING):
    return box_positions(origin,col,row,dep,particle_spacing)

def table_positions(origin=TABLE_ORIGIN,
                    particle_spacing=PARTICLE_SPACING,
                    table_top_col=TABLE_TOP_COL,
                    table_top_row=TABLE_TOP_ROW,
                    table_top_dep=TABLE_TOP_DEP,
                    table_leg_col=TABLE_LEG_COL,
                    table_leg_row=TABLE_LEG_ROW,
                    table_leg_dep=TABLE_LEG_DEP):
    top_origin=(origin[0],origin[1]+float(table_leg_row)*particle_spacing,origin[2])
    top=box_positions(top_origin,table_top_col,table_top_row,table_top_dep,particle_spacing)
    leg_offsets=[
        (0,0),
        (table_top_col-table_leg_col,0),
        (0,table_top_dep-table_leg_dep),
        (table_top_col-table_leg_col,table_top_dep-table_leg_dep),
    ]
    legs=[]
    for ox,oz in leg_offsets:
        leg_origin=(origin[0]+float(ox)*particle_spacing,origin[1],origin[2]+float(oz)*particle_spacing)
        legs.append(box_positions(leg_origin,table_leg_col,table_leg_row,table_leg_dep,particle_spacing))
    return np.concatenate([top]+legs,axis=0)

def table_model():
    positions=table_positions()
    positions=rotate_positions_z_clockwise(positions)
    labels=np.full(positions.shape[0],TABLE_LABEL,dtype=np.int32)
    return ModelSpec("table",positions,labels)

def cube_model(origin=CUBE_ORIGIN):
    positions=cube_positions(origin=origin)
    positions=rotate_positions_z_clockwise(positions)
    labels=np.full(positions.shape[0],CUBE_LABEL,dtype=np.int32)
    return ModelSpec("cube",positions,labels)

def model_from_name(name=DEFAULT_MODEL):
    if name=="cube":
        return cube_model()
    if name=="table":
        return table_model()
    raise ValueError("unknown model:"+str(name))

def resolve_model(model=None):
    if model is None:
        return model_from_name(DEFAULT_MODEL)
    if isinstance(model,str):
        return model_from_name(model)
    return model

def default_data_path(model=DEFAULT_MODEL):
    if not isinstance(model,str):
        model=model.name
    return DEFAULT_NPZ_TEMPLATE.format(model=model)

class SimState:
    def __init__(self,model=None,device_name=k.device):
        model=resolve_model(model)
        self.model=model
        self.device=device_name
        self.num_particles=model.num_particles
        self.grid_pos=wp.zeros(k.NUM_GRIDS,dtype=wp.vec3,device=device_name)
        self.grid_vel=wp.zeros(k.NUM_GRIDS,dtype=wp.vec3,device=device_name)
        self.grid_f=wp.zeros(k.NUM_GRIDS,dtype=wp.vec3,device=device_name)
        self.grid_mass=wp.zeros(k.NUM_GRIDS,dtype=float,device=device_name)
        self.particle_pos=wp.zeros(self.num_particles,dtype=wp.vec3,device=device_name)
        self.particle_vel=wp.zeros(self.num_particles,dtype=wp.vec3,device=device_name)
        self.particle_dvel=wp.zeros(self.num_particles,dtype=wp.mat33,device=device_name)
        self.particle_F=wp.zeros(self.num_particles,dtype=wp.mat33,device=device_name)
        self.particle_P=wp.zeros(self.num_particles,dtype=wp.mat33,device=device_name)
        self.particle_mass=wp.zeros(self.num_particles,dtype=float,device=device_name)
        self.particle_label=wp.zeros(self.num_particles,dtype=int,device=device_name)

def ensure_parent_dir(path):
    directory=os.path.dirname(path)
    if directory:
        os.makedirs(directory,exist_ok=True)

def init_state(state):
    wp.launch(
        k.init_grid,
        dim=k.NUM_GRIDS,
        inputs=[state.grid_pos,state.grid_vel,state.grid_f,state.grid_mass,k.GRID_SIZE,k.GRID_LEN,k.GRID_HEI],
        device=state.device,
    )
    initial_pos=wp.array(state.model.positions,dtype=wp.vec3,device=state.device)
    initial_label=wp.array(state.model.labels,dtype=int,device=state.device)
    wp.launch(
        k.init_particle_state,
        dim=state.num_particles,
        inputs=[
            initial_pos,
            initial_label,
            state.particle_pos,
            state.particle_vel,
            state.particle_dvel,
            state.particle_F,
            state.particle_P,
            state.particle_mass,
            state.particle_label,
            state.model.m0,
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
        dim=state.num_particles,
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
            k.GRID_DEP,
            state.model.v0,
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
        k.update_grid_vel,
        dim=k.NUM_GRIDS,
        inputs=[
            state.grid_vel,
            state.grid_f,
            state.grid_mass,
            wp.vec3(0.0,-k.GRAVITY,0.0),
            dt,
            k.GRID_SIZE,
            k.GRID_LEN,
            k.GRID_HEI,
            k.GRID_DEP,
            GRID_BOUNDARY,
        ],
        device=state.device,
    )
    wp.launch(
        k.apply_grid_ground_contact,
        dim=k.NUM_GRIDS,
        inputs=[
            state.grid_vel,
            state.grid_mass,
            k.GRID_SIZE,
            k.GRID_LEN,
            k.GRID_HEI,
            k.GROUND,
            k.RESTITUTION,
            k.DAMPING,
        ],
        device=state.device,
    )
    wp.launch(
        k.G2P,
        dim=state.num_particles,
        inputs=[
            state.grid_pos,
            state.grid_vel,
            state.particle_pos,
            state.particle_vel,
            state.particle_dvel,
            k.GRID_SIZE,
            k.GRID_LEN,
            k.GRID_HEI,
            k.GRID_DEP,
        ],
        device=state.device,
    )
    wp.launch(
        k.update_pos,
        dim=state.num_particles,
        inputs=[
            state.particle_pos,
            state.particle_vel,
            state.particle_dvel,
            state.particle_F,
            state.particle_P,
            k.GRID_SIZE,
            k.GRID_LEN,
            k.GRID_HEI,
            k.GRID_DEP,
            dt,
            k.GROUND,
            k.RESTITUTION,
            k.DAMPING,
            k.MU,
            k.LAMBDA,
        ],
        device=state.device,
    )

def warmup(model=None,device_name=k.device):
    state=SimState(model,device_name)
    init_state(state)
    substep(state,DEFAULT_FRAME_DT/float(DEFAULT_SUBSTEPS_PER_FRAME))
    wp.synchronize()

def particle_points(state):
    return state.particle_pos.numpy()

def particle_labels(state):
    return state.particle_label.numpy()

def sample_indices(num_particles,sample_count=DEFAULT_SAMPLE_COUNT,seed=0):
    sample_count=min(int(sample_count),int(num_particles))
    rng=np.random.default_rng(seed)
    return np.sort(rng.choice(int(num_particles),size=sample_count,replace=False)).astype(np.int32)

def sampled_particle_points(state,indices):
    return state.particle_pos.numpy()[indices]

def sampled_particle_labels(state,indices):
    return state.particle_label.numpy()[indices]

def domain_bounds():
    size=float(k.GRID_SIZE)
    return np.array([
        [0.0,0.0,0.0],
        [float(k.GRID_LEN-1)*size,float(k.GRID_HEI-1)*size,float(k.GRID_DEP-1)*size],
    ],dtype=np.float32)

def ground_square():
    bounds=domain_bounds()
    return np.array([
        [bounds[0,0],GROUND_Y,bounds[0,2]],
        [bounds[1,0],GROUND_Y,bounds[0,2]],
        [bounds[1,0],GROUND_Y,bounds[1,2]],
        [bounds[0,0],GROUND_Y,bounds[1,2]],
        [bounds[0,0],GROUND_Y,bounds[0,2]],
    ],dtype=np.float32)

def collect_sampled_frames(model=None,
                           num_frames=DEFAULT_NUM_FRAMES,
                           substeps_per_frame=DEFAULT_SUBSTEPS_PER_FRAME,
                           frame_dt=DEFAULT_FRAME_DT,
                           sample_count=DEFAULT_SAMPLE_COUNT,
                           sample_seed=0,
                           device_name=k.device):
    model=resolve_model(model)
    warmup(model,device_name)
    state=SimState(model,device_name)
    init_state(state)
    indices=sample_indices(state.num_particles,sample_count,sample_seed)
    labels=sampled_particle_labels(state,indices)
    frames=np.zeros((int(num_frames),indices.shape[0],3),dtype=np.float32)
    dt=frame_dt/float(substeps_per_frame)
    frames[0]=sampled_particle_points(state,indices)
    for frame in range(1,int(num_frames)):
        for _ in range(int(substeps_per_frame)):
            substep(state,dt)
        wp.synchronize()
        frames[frame]=sampled_particle_points(state,indices)
    return frames,labels,indices,model

def save_frames_npz(path=DEFAULT_NPZ_PATH,
                    model=None,
                    fps=DEFAULT_FPS,
                    num_frames=DEFAULT_NUM_FRAMES,
                    substeps_per_frame=DEFAULT_SUBSTEPS_PER_FRAME,
                    frame_dt=DEFAULT_FRAME_DT,
                    sample_count=DEFAULT_SAMPLE_COUNT,
                    sample_seed=0,
                    device_name=k.device):
    model=resolve_model(model)
    if path is None:
        path=default_data_path(model)
    ensure_parent_dir(path)
    frames,labels,indices,model=collect_sampled_frames(
        model=model,
        num_frames=num_frames,
        substeps_per_frame=substeps_per_frame,
        frame_dt=frame_dt,
        sample_count=sample_count,
        sample_seed=sample_seed,
        device_name=device_name,
    )
    np.savez(
        path,
        frames=frames,
        labels=labels,
        indices=indices,
        fps=np.int32(fps),
        grid_size=np.float32(k.GRID_SIZE),
        grid_shape=np.array([int(k.GRID_LEN),int(k.GRID_HEI),int(k.GRID_DEP)],dtype=np.int32),
        ground=np.float32(GROUND_Y),
        particles_per_grid=np.int32(k.PARTICLES_PER_GRID),
        total_particles=np.int32(model.num_particles),
        model=np.array(model.name),
    )
    return path

def build_parser():
    parser=argparse.ArgumentParser(description="Generate sampled 3D MPM frames.")
    parser.add_argument("--model",choices=["cube","table"],default=DEFAULT_MODEL)
    parser.add_argument("--path",default=None)
    parser.add_argument("--fps",type=int,default=DEFAULT_FPS)
    parser.add_argument("--num-frames",type=int,default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--substeps-per-frame",type=int,default=DEFAULT_SUBSTEPS_PER_FRAME)
    parser.add_argument("--frame-dt",type=float,default=DEFAULT_FRAME_DT)
    parser.add_argument("--sample-count",type=int,default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--sample-seed",type=int,default=0)
    parser.add_argument("--device",default=k.device)
    return parser

def main(argv=None):
    args=build_parser().parse_args(argv)
    model=model_from_name(args.model)
    path=args.path
    if path is None:
        path=default_data_path(args.model)
    path=save_frames_npz(
        path=path,
        model=model,
        fps=args.fps,
        num_frames=args.num_frames,
        substeps_per_frame=args.substeps_per_frame,
        frame_dt=args.frame_dt,
        sample_count=args.sample_count,
        sample_seed=args.sample_seed,
        device_name=args.device,
    )
    print(path)
    return path

if __name__=="__main__":
    main()
