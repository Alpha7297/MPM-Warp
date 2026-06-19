import argparse
import os
import sys

import numpy as np
import warp as wp

EXPERIMENT_DIR=os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
if EXPERIMENT_DIR not in sys.path:
    sys.path.insert(0,EXPERIMENT_DIR)

import generate as g
import kernels as k
from optimizer import AdamW

NUM_STEPS=300
SAVE_INTERNAL=200
START_EPOCH=600
END_EPOCH=1000
DT=g.DEFAULT_DT
LR_LIST=[(0,1e-1),(50,1e-1)]
POSITION_SCALE=1000.0
BETA1=0.9
BETA2=0.999
EPSILON=1.0e-15
WEIGHT_DECAY=0.0
GROUND_TRUTH_SPEED=1.0
TRAIN_INITIAL_SPEED=1e-2
TRAIN_RANDOM_SEED=123654
NET_DIR=os.path.join(os.path.dirname(__file__),"net")

@wp.kernel
def build_initial_velocity(velocity_values:wp.array(dtype=float),
                           initial_velocity:wp.array(dtype=wp.vec2)):
    i=wp.tid()
    initial_velocity[i]=wp.vec2(velocity_values[i*2],velocity_values[i*2+1])

@wp.kernel
def final_position_loss(particle_pos:wp.array(dtype=wp.vec2),
                        target_pos:wp.array(dtype=wp.vec2),
                        loss:wp.array(dtype=float),
                        num_particles:int,
                        final_step:int,
                        position_scale:float,
                        loss_scale:float):
    i=wp.tid()
    idx=final_step*num_particles+i
    error=(particle_pos[idx]-target_pos[i])*position_scale
    wp.atomic_add(loss,0,wp.dot(error,error)*loss_scale)

def model_initial_positions(model):
    return model.positions.astype(np.float32).copy()

def uniform_initial_velocity(model,velocity):
    initial_velocity=np.zeros((model.num_particles,2),dtype=np.float32)
    initial_velocity[:]=np.asarray(velocity,dtype=np.float32)
    return initial_velocity

def symmetric_initial_velocity(model,speed=GROUND_TRUTH_SPEED):
    positions=model_initial_positions(model)
    center=positions.mean(axis=0)
    direction=np.zeros_like(positions,dtype=np.float32)
    direction[:,0]=np.sign(positions[:,0]-center[0])
    direction[:,1]=-np.sign(positions[:,1]-center[1])
    norm=np.linalg.norm(direction,axis=1)
    mask=norm>0.0
    direction[mask]=direction[mask]/norm[mask,None]
    return direction.astype(np.float32)*float(speed)

def sin_cos_initial_velocity(model,speed=GROUND_TRUTH_SPEED):
    positions=model_initial_positions(model)
    minimum=positions.min(axis=0)
    maximum=positions.max(axis=0)
    extent=maximum-minimum
    if np.any(extent<=0.0):
        raise ValueError("model extent must be positive")
    center=0.5*(minimum+maximum)
    phase_x=(positions[:,0]-center[0])*(np.pi/extent[0])
    phase_y=(positions[:,1]-center[1])*(np.pi/extent[1])
    velocity=np.zeros_like(positions,dtype=np.float32)
    velocity[:,0]=float(speed)*np.sin(phase_x)*np.cos(phase_y)+1.0
    velocity[:,1]=-float(speed)*np.cos(phase_x)*np.sin(phase_y)
    return velocity

def random_initial_velocity(model,speed=TRAIN_INITIAL_SPEED,seed=TRAIN_RANDOM_SEED):
    rng=np.random.default_rng(seed)
    direction=rng.uniform(-1.0,1.0,size=(model.num_particles,2)).astype(np.float32)
    norm=np.linalg.norm(direction,axis=1)
    mask=norm>0.0
    direction[mask]=direction[mask]/norm[mask,None]
    magnitude=rng.uniform(0.0,float(speed),size=(model.num_particles,1)).astype(np.float32)
    return direction*magnitude

def ensure_parent_dir(path):
    directory=os.path.dirname(path)
    if directory:
        os.makedirs(directory,exist_ok=True)

def checkpoint_path(model_id,epoch,net_dir=NET_DIR):
    return os.path.join(net_dir,f"{int(model_id)}_{int(epoch)}.npz")

def checkpoint_epoch_from_name(name,model_id):
    prefix=f"{int(model_id)}_"
    suffix=".npz"
    if not name.startswith(prefix) or not name.endswith(suffix):
        return None
    try:
        return int(name[len(prefix):-len(suffix)])
    except ValueError:
        return None

def latest_checkpoint_path(model_id,net_dir=NET_DIR):
    if not os.path.isdir(net_dir):
        return None
    best_epoch=None
    best_path=None
    for name in os.listdir(net_dir):
        epoch=checkpoint_epoch_from_name(name,model_id)
        if epoch is None:
            continue
        path=os.path.join(net_dir,name)
        try:
            with np.load(path) as data:
                if "ground_truth_velocity" not in data.files or "optimizer_step" not in data.files:
                    continue
        except Exception:
            continue
        if best_epoch is None or epoch>best_epoch:
            best_epoch=epoch
            best_path=path
    return best_path

def save_checkpoint(path,model_id,epoch,loss_value,velocity_values,ground_truth_velocity_values,target_final_pos,history,optimizer):
    ensure_parent_dir(path)
    optimizer_state=optimizer.state_dict()
    payload={
        "model_id":int(model_id),
        "epoch":int(epoch),
        "loss":float(loss_value),
        "velocity":np.asarray(velocity_values,dtype=np.float32),
        "ground_truth_velocity":np.asarray(ground_truth_velocity_values,dtype=np.float32),
        "target":np.asarray(target_final_pos,dtype=np.float32),
        "history":np.asarray(history,dtype=np.float32),
        "optimizer_step":int(optimizer_state["step_count"]),
        "optimizer_beta1":float(optimizer_state["beta1"]),
        "optimizer_beta2":float(optimizer_state["beta2"]),
        "optimizer_epsilon":float(optimizer_state["epsilon"]),
        "optimizer_weight_decay":float(optimizer_state["weight_decay"]),
        "optimizer_parameter_count":len(optimizer_state["first_moments"]),
    }
    for i,(first_moment,second_moment) in enumerate(zip(optimizer_state["first_moments"],optimizer_state["second_moments"])):
        payload[f"optimizer_first_moment_{i}"]=first_moment
        payload[f"optimizer_second_moment_{i}"]=second_moment
    np.savez(path,**payload)
    return path

def load_checkpoint(path):
    with np.load(path) as data:
        velocity=data["velocity"].astype(np.float32).copy()
        epoch=int(data["epoch"]) if "epoch" in data.files else -1
        loss=float(data["loss"]) if "loss" in data.files else float("nan")
    return velocity,epoch,loss

def load_training_checkpoint(path):
    with np.load(path) as data:
        optimizer_keys={
            "ground_truth_velocity",
            "optimizer_step",
            "optimizer_beta1",
            "optimizer_beta2",
            "optimizer_epsilon",
            "optimizer_weight_decay",
            "optimizer_parameter_count",
        }
        if not optimizer_keys.issubset(data.files):
            raise ValueError("checkpoint does not contain ground truth and AdamW state; restart training from epoch 0")
        parameter_count=int(data["optimizer_parameter_count"])
        first_moments=[]
        second_moments=[]
        for i in range(parameter_count):
            first_key=f"optimizer_first_moment_{i}"
            second_key=f"optimizer_second_moment_{i}"
            if first_key not in data.files or second_key not in data.files:
                raise ValueError("checkpoint contains incomplete AdamW state")
            first_moments.append(data[first_key].astype(np.float32).copy())
            second_moments.append(data[second_key].astype(np.float32).copy())
        return {
            "model_id":int(data["model_id"]),
            "epoch":int(data["epoch"]),
            "loss":float(data["loss"]),
            "velocity":data["velocity"].astype(np.float32).copy(),
            "ground_truth_velocity":data["ground_truth_velocity"].astype(np.float32).copy(),
            "target":data["target"].astype(np.float32).copy(),
            "history":data["history"].astype(np.float32).tolist(),
            "optimizer":{
                "step_count":int(data["optimizer_step"]),
                "beta1":float(data["optimizer_beta1"]),
                "beta2":float(data["optimizer_beta2"]),
                "epsilon":float(data["optimizer_epsilon"]),
                "weight_decay":float(data["optimizer_weight_decay"]),
                "first_moments":first_moments,
                "second_moments":second_moments,
            },
        }

def case_initial_conditions(model_id):
    model_id=int(model_id)
    model=g.rectangle_model()
    if model_id==1:
        ground_truth_velocity=uniform_initial_velocity(model,np.array([GROUND_TRUTH_SPEED,0.0],dtype=np.float32))
        gravity_y=-float(k.GRAVITY)
    elif model_id==2:
        ground_truth_velocity=symmetric_initial_velocity(model,GROUND_TRUTH_SPEED)
        gravity_y=0.0
    elif model_id==3:
        ground_truth_velocity=sin_cos_initial_velocity(model,GROUND_TRUTH_SPEED)
        gravity_y=-float(k.GRAVITY)
    else:
        raise ValueError("model must be 1,2,or 3")
    initial_velocity=random_initial_velocity(model,TRAIN_INITIAL_SPEED,TRAIN_RANDOM_SEED)
    return model,ground_truth_velocity,initial_velocity,gravity_y

def should_save_checkpoint(epoch,end_epoch,save_internal):
    if epoch==end_epoch:
        return True
    return save_internal>0 and epoch%save_internal==0

def value_at_epoch(schedule,epoch):
    if not schedule or schedule[0][0]!=0:
        raise ValueError("learning rate schedule must start at epoch 0")
    value=schedule[0][1]
    previous_epoch=-1
    for start_epoch,candidate in schedule:
        if start_epoch<=previous_epoch:
            raise ValueError("learning rate schedule epochs must be strictly increasing")
        if candidate<=0.0:
            raise ValueError("learning rates must be positive")
        if epoch>=start_epoch:
            value=candidate
        previous_epoch=start_epoch
    return float(value)

def lr_at_epoch(epoch,lr_list=LR_LIST):
    return value_at_epoch(lr_list,epoch)

def rollout(state,num_steps,dt,gravity_y):
    for t in range(num_steps):
        wp.launch(
            k.theory_strain,
            dim=state.num_particles,
            inputs=[state.particle_dvel,state.particle_F,state.particle_P,t,state.num_particles,dt,k.MU,k.LAMBDA],
            device=state.device,
        )
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
            inputs=[
                state.grid_momentum,
                state.grid_f,
                state.grid_mass,
                state.grid_vel,
                t,
                int(k.NUM_GRIDS),
                wp.vec2(0.0,gravity_y),
                dt,
            ],
            device=state.device,
        )
        wp.launch(
            k.G2P,
            dim=state.num_particles,
            inputs=[
                state.grid_vel,
                state.particle_pos,
                state.particle_vel_trial,
                state.particle_dvel,
                t,
                state.num_particles,
                int(k.NUM_GRIDS),
                k.GRID_SIZE,
                k.GRID_LEN,
                k.GRID_HEI,
            ],
            device=state.device,
        )
        wp.launch(
            k.update_pos,
            dim=state.num_particles,
            inputs=[
                state.particle_pos,
                state.particle_vel,
                state.particle_vel_trial,
                t,
                state.num_particles,
                dt,
                g.GROUND,
                g.RESTITUTION,
                g.DAMPING,
            ],
            device=state.device,
        )

def collect_target_final_pos(model,ground_truth_velocity_values,gravity_y,num_steps=NUM_STEPS,dt=DT,device_name=None):
    if device_name is None:
        device_name=g.default_device_name()
    ground_truth_velocity_values=np.asarray(ground_truth_velocity_values,dtype=np.float32)
    if ground_truth_velocity_values.shape!=(model.num_particles,2):
        raise ValueError("ground_truth_velocity_values shape must be (num_particles,2)")
    state=g.SimState(model,num_steps,device_name,requires_grad=False)
    initial_velocity=g.initial_velocity_array(model,device_name,ground_truth_velocity_values)
    g.init_state(state,initial_velocity)
    rollout(state,num_steps,dt,gravity_y)
    wp.synchronize()
    final_offset=num_steps*model.num_particles
    return state.particle_pos.numpy()[final_offset:final_offset+model.num_particles].copy()

def case_data(model_id,num_steps=NUM_STEPS,dt=DT,device_name=None):
    model,ground_truth_velocity,initial_velocity,gravity_y=case_initial_conditions(model_id)
    target=collect_target_final_pos(model,ground_truth_velocity,gravity_y,num_steps,dt,device_name)
    return model,target,ground_truth_velocity,initial_velocity,gravity_y

def train_to_target(target_final_pos,
                    ground_truth_velocity_values,
                    initial_velocity_values,
                    gravity_y=-float(k.GRAVITY),
                    model=None,
                    num_steps=NUM_STEPS,
                    start_epoch=START_EPOCH,
                    end_epoch=END_EPOCH,
                    save_internal=SAVE_INTERNAL,
                    dt=DT,
                    lr_list=LR_LIST,
                    device_name=None,
                    model_id=1,
                    net_dir=NET_DIR):
    if model is None:
        model=g.rectangle_model()
    if device_name is None:
        device_name=g.default_device_name()
    target_final_pos=np.asarray(target_final_pos,dtype=np.float32)
    if target_final_pos.shape!=(model.num_particles,2):
        raise ValueError("target_final_pos shape must be (num_particles,2)")
    ground_truth_velocity_values=np.asarray(ground_truth_velocity_values,dtype=np.float32)
    initial_velocity_values=np.asarray(initial_velocity_values,dtype=np.float32)
    checkpoint=None
    history=[]
    if start_epoch!=0:
        path=checkpoint_path(model_id,start_epoch,net_dir)
        try:
            checkpoint=load_training_checkpoint(path)
        except Exception as exc:
            print(f"failed to load checkpoint {path}: {exc}")
            sys.exit(1)
        if checkpoint["model_id"]!=int(model_id):
            raise ValueError("checkpoint model_id does not match")
        if checkpoint["epoch"]!=int(start_epoch):
            raise ValueError("checkpoint epoch does not match start_epoch")
        if checkpoint["ground_truth_velocity"].shape!=ground_truth_velocity_values.shape or not np.allclose(checkpoint["ground_truth_velocity"],ground_truth_velocity_values):
            raise ValueError("checkpoint ground truth velocity does not match")
        if checkpoint["target"].shape!=target_final_pos.shape or not np.allclose(checkpoint["target"],target_final_pos):
            raise ValueError("checkpoint target does not match current target")
        initial_velocity_values=checkpoint["velocity"]
        history=checkpoint["history"]
        print(f"loaded={path} epoch={checkpoint['epoch']} loss={checkpoint['loss']:.6e}")
    if ground_truth_velocity_values.shape!=(model.num_particles,2):
        raise ValueError("ground_truth_velocity_values shape must be (num_particles,2)")
    if initial_velocity_values.shape!=(model.num_particles,2):
        raise ValueError("initial_velocity_values shape must be (num_particles,2)")
    state=g.SimState(model,num_steps,device_name,requires_grad=True)
    velocity_values=wp.array(initial_velocity_values.reshape(-1).copy(),dtype=float,device=device_name,requires_grad=True)
    initial_velocity=wp.zeros(model.num_particles,dtype=wp.vec2,device=device_name,requires_grad=True)
    target=wp.array(target_final_pos,dtype=wp.vec2,device=device_name)
    loss=wp.zeros(1,dtype=float,device=device_name,requires_grad=True)
    optimizer=AdamW(
        [velocity_values],
        beta1=BETA1,
        beta2=BETA2,
        epsilon=EPSILON,
        weight_decay=WEIGHT_DECAY,
    )
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    loss_scale=1.0/float(model.num_particles)
    final_checkpoint=None
    start=start_epoch+1 if start_epoch!=0 else 0

    for epoch in range(start,end_epoch+1):
        lr=lr_at_epoch(epoch,lr_list)
        state.zero_forward()
        state.zero_grad()
        initial_velocity.zero_()
        initial_velocity.grad.zero_()
        loss.zero_()
        loss.grad.zero_()
        optimizer.zero_grad()
        tape=wp.Tape()
        with tape:
            wp.launch(
                build_initial_velocity,
                dim=model.num_particles,
                inputs=[velocity_values,initial_velocity],
                device=device_name,
            )
            g.init_state(state,initial_velocity)
            rollout(state,num_steps,dt,gravity_y)
            wp.launch(
                final_position_loss,
                dim=model.num_particles,
                inputs=[state.particle_pos,target,loss,state.num_particles,num_steps,POSITION_SCALE,loss_scale],
                device=device_name,
            )
        tape.backward(loss)
        optimizer.step(lr)
        wp.synchronize()
        loss_value=float(loss.numpy()[0])
        history.append(loss_value)
        if should_save_checkpoint(epoch,end_epoch,save_internal):
            final_checkpoint=save_checkpoint(
                checkpoint_path(model_id,epoch,net_dir),
                model_id,
                epoch,
                loss_value,
                velocity_values.numpy().reshape(model.num_particles,2),
                ground_truth_velocity_values,
                target_final_pos,
                history,
                optimizer,
            )
        print(f"epoch={epoch} lr={lr:.6e} loss={loss_value:.6e}")

    return {
        "velocity":velocity_values.numpy().reshape(model.num_particles,2),
        "ground_truth_velocity":ground_truth_velocity_values,
        "loss":history,
        "target":target_final_pos,
        "checkpoint":final_checkpoint,
    }

def train_model(model_id,start_epoch=START_EPOCH,end_epoch=END_EPOCH,lr_list=LR_LIST,device_name=None):
    model,ground_truth_velocity,initial_velocity,gravity_y=case_initial_conditions(model_id)
    if start_epoch==0:
        target=collect_target_final_pos(model,ground_truth_velocity,gravity_y,NUM_STEPS,DT,device_name)
    else:
        path=checkpoint_path(model_id,start_epoch,NET_DIR)
        try:
            with np.load(path) as data:
                target=data["target"].astype(np.float32).copy()
                checkpoint_ground_truth=data["ground_truth_velocity"].astype(np.float32).copy()
        except Exception as exc:
            print(f"failed to load supervision from checkpoint {path}: {exc}")
            sys.exit(1)
        if checkpoint_ground_truth.shape!=ground_truth_velocity.shape or not np.allclose(checkpoint_ground_truth,ground_truth_velocity):
            raise ValueError("checkpoint ground truth velocity does not match current case")
        ground_truth_velocity=checkpoint_ground_truth
    return train_to_target(
        target,
        ground_truth_velocity,
        initial_velocity,
        gravity_y,
        model,
        NUM_STEPS,
        start_epoch,
        end_epoch,
        SAVE_INTERNAL,
        DT,
        lr_list,
        device_name,
        model_id,
        NET_DIR,
    )

def train_free(start_epoch=START_EPOCH,end_epoch=END_EPOCH,lr_list=LR_LIST,device_name=None):
    return train_model(1,start_epoch,end_epoch,lr_list,device_name)

def train_no_gravity(start_epoch=START_EPOCH,end_epoch=END_EPOCH,lr_list=LR_LIST,device_name=None):
    return train_model(2,start_epoch,end_epoch,lr_list,device_name)

def train_combined(start_epoch=START_EPOCH,end_epoch=END_EPOCH,lr_list=LR_LIST,device_name=None):
    return train_model(3,start_epoch,end_epoch,lr_list,device_name)

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",type=int,default=1,choices=[1,2,3])
    return parser.parse_args()

def main():
    args=parse_args()
    return train_model(args.model)

if __name__=="__main__":
    main()
