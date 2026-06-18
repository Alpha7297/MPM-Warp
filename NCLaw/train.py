import os

import torch
import warp as wp

try:
    from . import kernels as k
    from . import generate as g
    from .layers import Net
except ImportError:
    import kernels as k
    import generate as g
    from layers import Net

STEPS_LIST=[(0,20),(1000,100),(2000,200)]
LR_LIST=[(0,1.0e-2),(500,1.0e-3),(1000,1.0e-4)]
LOSS_SUBSTEPS=5
START_EPOCH=3000
END_EPOCH=5000
DT=5.0e-4
NET_DIR=os.path.join(os.path.dirname(__file__),"net")
TRAIN_MODEL="rectangle"
VELOCITY_RANGE=0.1

@wp.kernel
def init_random_velocity(particle_vel:wp.array(dtype=wp.vec2),
                         random_vel:wp.array(dtype=wp.vec2)):
    i=wp.tid()
    particle_vel[i]=random_vel[i]

def default_device_name():
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"

def torch_device(device_name):
    if device_name.startswith("cuda"):
        return torch.device(device_name)
    return torch.device("cpu")

def choose_model():
    if TRAIN_MODEL=="rectangle":
        return g.rectangle_model()
    if TRAIN_MODEL=="table":
        return g.table_model()
    raise ValueError(f"unknown TRAIN_MODEL={TRAIN_MODEL}")

def init_train_velocity(state):
    random_vel=(torch.rand((state.num_particles,2),dtype=torch.float32)*2.0-1.0)*VELOCITY_RANGE
    random_vel_wp=wp.array(random_vel.numpy(),dtype=wp.vec2,device=state.device)
    wp.launch(
        init_random_velocity,
        dim=state.num_particles,
        inputs=[state.particle_vel,random_vel_wp],
        device=state.device,
    )

def substep_theory(state,dt):
    wp.launch(
        k.theory_strain,
        dim=state.num_particles,
        inputs=[state.particle_dvel,state.particle_F,state.particle_P,dt,k.MU,k.LAMBDA],
        device=state.device,
    )
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
        k.copy_vel,
        dim=k.NUM_GRIDS,
        inputs=[state.grid_vel,state.grid_vel_old],
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
        dim=state.num_particles,
        inputs=[
            state.grid_pos,
            state.grid_vel,
            state.grid_vel_old,
            state.particle_pos,
            state.particle_vel,
            state.particle_dvel,
            k.GRID_SIZE,
            k.GRID_LEN,
            k.GRID_HEI,
            k.FLIP_RATIO,
        ],
        device=state.device,
    )
    wp.launch(
        k.update_pos,
        dim=state.num_particles,
        inputs=[state.particle_pos,state.particle_vel,state.grid_idx,k.GRID_SIZE,k.GRID_LEN,dt],
        device=state.device,
    )

def collect_batch(device_name,total_steps):
    model=choose_model()
    state=g.SimState(model,device_name)
    g.init_state(state)
    init_train_velocity(state)
    F_list=[]
    P_list=[]
    for step in range(1,total_steps+1):
        substep_theory(state,DT)
        if step%LOSS_SUBSTEPS==0 or step==total_steps:
            wp.synchronize()
            F_list.append(wp.to_torch(state.particle_F).detach().clone())
            P_list.append(wp.to_torch(state.particle_P).detach().clone())
    wp.synchronize()
    return torch.stack(F_list,dim=0),torch.stack(P_list,dim=0)

def loss_fn(net,F_batch,P_batch):
    pred=net(F_batch.reshape(-1,2,2)).reshape_as(P_batch)
    loss_per_time=torch.mean((pred-P_batch)**2,dim=(1,2,3))
    return torch.mean(loss_per_time)

def save_net(net,epoch,loss):
    os.makedirs(NET_DIR,exist_ok=True)
    path=os.path.join(NET_DIR,f"{epoch}.plt")
    torch.save({"epoch":epoch,"loss":float(loss),"state_dict":net.state_dict()},path)
    return path

def load_net(net,epoch,device):
    path=os.path.join(NET_DIR,f"{epoch}.plt")
    checkpoint=torch.load(path,map_location=device)
    net.load_state_dict(checkpoint["state_dict"])
    return checkpoint

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

def train():
    device_name=default_device_name()
    device=torch_device(device_name)
    current_steps=steps_at_epoch(START_EPOCH)
    F_batch,P_batch=collect_batch(device_name,current_steps)
    F_batch=F_batch.to(device)
    P_batch=P_batch.to(device)
    net=Net().to(device)
    if START_EPOCH!=0:
        checkpoint=load_net(net,START_EPOCH,device)
        print(f"loaded={os.path.join(NET_DIR,f'{START_EPOCH}.plt')} loss={checkpoint.get('loss')}")
    optimizer=torch.optim.AdamW(net.parameters(),lr=lr_at_epoch(START_EPOCH+1))
    start=0
    if START_EPOCH!=0:
        start=START_EPOCH+1
    for epoch in range(start,END_EPOCH+1):
        total_steps=steps_at_epoch(epoch)
        if total_steps!=current_steps:
            current_steps=total_steps
            F_batch,P_batch=collect_batch(device_name,current_steps)
            F_batch=F_batch.to(device)
            P_batch=P_batch.to(device)
        lr=lr_at_epoch(epoch)
        for group in optimizer.param_groups:
            group["lr"]=lr
        optimizer.zero_grad()
        loss=loss_fn(net,F_batch,P_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),0.1)
        optimizer.step()
        if epoch%500==0:
            path=save_net(net,epoch,loss.detach().cpu())
        print(f"epoch={epoch} steps={current_steps} lr={lr:.6e} loss={float(loss.detach().cpu()):.6e}")

if __name__=="__main__":
    train()
