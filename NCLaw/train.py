import os

import numpy as np
import warp as wp

import generate as g
import kernels as k
from MLP import AdamW,MLP

STEPS_LIST=[(0,20),(1000,100),(2000,200)]
LR_LIST=[(0,1.0e-1),(500,1.0e-2),(1000,1.0e-3)]
LOSS_SUBSTEPS=5
START_EPOCH=0
END_EPOCH=3000
DT=5.0e-4
NET_DIR=os.path.join(os.path.dirname(__file__),"net")
TRAIN_MODEL="rectangle"
VELOCITY_RANGE=0.1
POSITION_SCALE=1000.0
BETA1=0.9
BETA2=0.999
EPSILON=1.0e-15
WEIGHT_DECAY=1.0e-2
RANDOM_SEED=0

def default_device_name():
    return g.default_device_name()

def choose_model():
    if TRAIN_MODEL=="rectangle":
        return g.rectangle_model()
    if TRAIN_MODEL=="table":
        return g.table_model()
    raise ValueError(f"unknown TRAIN_MODEL={TRAIN_MODEL}")

def value_at_epoch(schedule,epoch):
    value=schedule[0][1]
    for start_epoch,candidate in schedule:
        if epoch>=start_epoch:
            value=candidate
        else:
            break
    return value

def steps_at_epoch(epoch):
    return value_at_epoch(STEPS_LIST,epoch)

def lr_at_epoch(epoch):
    return value_at_epoch(LR_LIST,epoch)

def is_loss_step(step,total_steps):
    return step%LOSS_SUBSTEPS==0 or step==total_steps

def random_velocity(model,device_name,rng):
    values=rng.uniform(-VELOCITY_RANGE,VELOCITY_RANGE,size=(model.num_particles,2)).astype(np.float32)
    return wp.array(values,dtype=wp.vec2,device=device_name)

def rollout(state,total_steps,mode,net=None,target_pos=None,loss=None):
    loss_steps=sum(1 for step in range(1,total_steps+1) if is_loss_step(step,total_steps))
    loss_scale=1.0/float(loss_steps*state.num_particles)
    for t in range(total_steps):
        g.substep(state,DT,mode,t,net)
        if loss is not None and is_loss_step(t+1,total_steps):
            wp.launch(
                k.position_loss,
                dim=state.num_particles,
                inputs=[state.particle_pos,target_pos,loss,t,state.num_particles,POSITION_SCALE,loss_scale],
                device=state.device,
            )

def collect_target(model,total_steps,device_name,initial_velocity):
    target_state=g.SimState(model,total_steps,device_name,requires_grad=False)
    g.init_state(target_state,initial_velocity)
    rollout(target_state,total_steps,"tradition")
    wp.synchronize()
    return target_state

def prepare_training_state(state,net,initial_velocity):
    state.zero_forward()
    state.zero_grad()
    net.zero_workspace()
    g.init_state(state,initial_velocity)

def checkpoint_path(epoch):
    return os.path.join(NET_DIR,f"{epoch}.npz")

def train():
    device_name=default_device_name()
    model=choose_model()
    max_steps=steps_at_epoch(END_EPOCH)
    rng=np.random.default_rng(RANDOM_SEED)
    initial_velocity=random_velocity(model,device_name,rng)
    target_state=collect_target(model,max_steps,device_name,initial_velocity)
    state=g.SimState(model,max_steps,device_name,requires_grad=True)
    net=MLP(model.num_particles,max_steps,device_name,seed=RANDOM_SEED)
    if START_EPOCH!=0:
        loaded_epoch,loaded_loss=net.load(checkpoint_path(START_EPOCH))
        print(f"loaded={checkpoint_path(loaded_epoch)} loss={loaded_loss}")
    optimizer=AdamW(
        net.parameters,
        beta1=BETA1,
        beta2=BETA2,
        epsilon=EPSILON,
        weight_decay=WEIGHT_DECAY,
    )
    loss=wp.zeros(1,dtype=float,device=device_name,requires_grad=True)
    start=0
    if START_EPOCH!=0:
        start=START_EPOCH+1

    for epoch in range(start,END_EPOCH+1):
        total_steps=steps_at_epoch(epoch)
        lr=lr_at_epoch(epoch)
        optimizer.zero_grad()
        prepare_training_state(state,net,initial_velocity)
        loss.zero_()
        tape=wp.Tape()
        with tape:
            rollout(state,total_steps,"nclaw",net,target_state.particle_pos,loss)
        tape.backward(loss)
        optimizer.step(lr)
        wp.synchronize()
        loss_value=float(loss.numpy()[0])
        if epoch%500==0:
            os.makedirs(NET_DIR,exist_ok=True)
            net.save(checkpoint_path(epoch),epoch,loss_value)
        print(f"epoch={epoch} steps={total_steps} lr={lr:.6e} position_loss={loss_value:.6e}")

if __name__=="__main__":
    train()
