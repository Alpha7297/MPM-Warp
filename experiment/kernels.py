import os

import warp as wp

wp.config.kernel_cache_dir=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","outputs","warp_cache"))
wp.init()
device="cuda:0"

MODULUS=wp.constant(20000.0)
POISSON=wp.constant(0.2)
MU=wp.constant(MODULUS/(2.0*(1.0+POISSON)))
LAMBDA=wp.constant(MODULUS*POISSON/((1.0+POISSON)*(1.0-2.0*POISSON)))
GRAVITY=wp.constant(9.8)

GRID_SIZE=wp.constant(0.014)
GRID_LEN=wp.constant(160)
GRID_HEI=wp.constant(160)
NUM_GRIDS=wp.constant(GRID_LEN*GRID_HEI)

@wp.func
def kernel_func(r:float):
    ar=wp.abs(r)
    if ar<0.5:
        return 0.75-ar*ar
    elif ar<1.5:
        value=1.5-ar
        return 0.5*value*value
    else:
        return 0.0

@wp.func
def dkernel_func(r:float):
    if r>-0.5 and r<0.5:
        return -2.0*r
    elif r>=0.5 and r<1.5:
        return r-1.5
    elif r>-1.5 and r<=-0.5:
        return r+1.5
    else:
        return 0.0

@wp.kernel
def init_particle_state(initial_pos:wp.array(dtype=wp.vec2),
                        initial_vel:wp.array(dtype=wp.vec2),
                        particle_pos:wp.array(dtype=wp.vec2),
                        particle_vel:wp.array(dtype=wp.vec2),
                        particle_dvel:wp.array(dtype=wp.mat22),
                        particle_F:wp.array(dtype=wp.mat22),
                        particle_P:wp.array(dtype=wp.mat22),
                        t:int,
                        num_particles:int):
    i=wp.tid()
    offset=t*num_particles+i
    particle_pos[offset]=initial_pos[i]
    particle_vel[offset]=initial_vel[i]
    particle_dvel[offset]=wp.mat22(0.0,0.0,0.0,0.0)
    particle_F[offset]=wp.mat22(1.0,0.0,0.0,1.0)
    particle_P[offset]=wp.mat22(0.0,0.0,0.0,0.0)

@wp.kernel
def theory_strain(particle_dvel:wp.array(dtype=wp.mat22),
                  particle_F:wp.array(dtype=wp.mat22),
                  particle_P:wp.array(dtype=wp.mat22),
                  t:int,
                  num_particles:int,
                  dt:float,
                  mu:float,
                  lbd:float):
    i=wp.tid()
    current=t*num_particles+i
    next=(t+1)*num_particles+i
    F=particle_F[current]+(particle_dvel[current]@particle_F[current])*dt
    J=wp.determinant(F)
    LogJ=wp.log(J)
    particle_F[next]=F

    p00=mu*F[0,0]-F[1,1]*(mu-lbd*LogJ)/J
    p01=mu*F[0,1]+F[1,0]*(mu-lbd*LogJ)/J
    p10=mu*F[1,0]+F[0,1]*(mu-lbd*LogJ)/J
    p11=mu*F[1,1]-F[0,0]*(mu-lbd*LogJ)/J
    particle_P[next]=wp.mat22(p00,p01,p10,p11)

@wp.kernel
def update_F(particle_dvel:wp.array(dtype=wp.mat22),
             particle_F:wp.array(dtype=wp.mat22),
             t:int,
             num_particles:int,
             dt:float):
    i=wp.tid()
    current=t*num_particles+i
    next=(t+1)*num_particles+i
    particle_F[next]=particle_F[current]+(particle_dvel[current]@particle_F[current])*dt

@wp.kernel
def P2G_update_grid(grid_momentum:wp.array(dtype=wp.vec2),
                    grid_f:wp.array(dtype=wp.vec2),
                    grid_mass:wp.array(dtype=float),
                    particle_pos:wp.array(dtype=wp.vec2),
                    particle_vel:wp.array(dtype=wp.vec2),
                    particle_F:wp.array(dtype=wp.mat22),
                    particle_P:wp.array(dtype=wp.mat22),
                    t:int,
                    num_particles:int,
                    num_grids:int,
                    particle_mass:float,
                    grid_size:float,
                    grid_wid:int,
                    grid_hei:int,
                    V0:float):
    i=wp.tid()
    particle_current=t*num_particles+i
    particle_next=(t+1)*num_particles+i
    grid_offset=t*num_grids
    pos=particle_pos[particle_current]
    base_x=int(pos[0]/grid_size-0.5)
    base_y=int(pos[1]/grid_size-0.5)
    for ox in range(3):
        for oy in range(3):
            gx=base_x+ox
            gy=base_y+oy
            if gx>=0 and gx<grid_wid and gy>=0 and gy<grid_hei:
                idx=grid_offset+gy*grid_wid+gx
                grid_pos=wp.vec2(float(gx)*grid_size,float(gy)*grid_size)
                r=pos-grid_pos
                wx=kernel_func(r[0]/grid_size)
                wy=kernel_func(r[1]/grid_size)
                weight=wx*wy
                wp.atomic_add(grid_momentum,idx,particle_vel[particle_current]*(particle_mass*weight))
                wp.atomic_add(grid_mass,idx,particle_mass*weight)

                dN=wp.vec2(dkernel_func(r[0]/grid_size)*wy,dkernel_func(r[1]/grid_size)*wx)
                PFt=particle_P[particle_next]@wp.transpose(particle_F[particle_next])
                wp.atomic_add(grid_f,idx,(PFt@dN)*(-V0/grid_size))

@wp.kernel
def update_grid_vel(grid_momentum:wp.array(dtype=wp.vec2),
                    grid_f:wp.array(dtype=wp.vec2),
                    grid_mass:wp.array(dtype=float),
                    grid_vel:wp.array(dtype=wp.vec2),
                    t:int,
                    num_grids:int,
                    f_extern:wp.vec2,
                    dt:float):
    i=wp.tid()
    idx=t*num_grids+i
    mass=grid_mass[idx]
    if mass>0.0:
        velocity=grid_momentum[idx]/mass
        grid_vel[idx]=velocity+dt*(grid_f[idx]/mass+f_extern)
    else:
        grid_vel[idx]=wp.vec2(0.0,0.0)

@wp.kernel
def G2P(grid_vel:wp.array(dtype=wp.vec2),
        particle_pos:wp.array(dtype=wp.vec2),
        particle_vel_trial:wp.array(dtype=wp.vec2),
        particle_dvel:wp.array(dtype=wp.mat22),
        t:int,
        num_particles:int,
        num_grids:int,
        grid_size:float,
        grid_wid:int,
        grid_hei:int):
    i=wp.tid()
    particle_current=t*num_particles+i
    particle_next=(t+1)*num_particles+i
    grid_offset=t*num_grids
    pos=particle_pos[particle_current]
    pic_v=wp.vec2(0.0,0.0)
    dvel=wp.mat22(0.0,0.0,0.0,0.0)
    base_x=int(pos[0]/grid_size-0.5)
    base_y=int(pos[1]/grid_size-0.5)

    for ox in range(3):
        for oy in range(3):
            gx=base_x+ox
            gy=base_y+oy
            if gx>=0 and gx<grid_wid and gy>=0 and gy<grid_hei:
                idx=grid_offset+gy*grid_wid+gx
                grid_pos=wp.vec2(float(gx)*grid_size,float(gy)*grid_size)
                r=pos-grid_pos
                wx=kernel_func(r[0]/grid_size)
                wy=kernel_func(r[1]/grid_size)
                weight=wx*wy
                dN=wp.vec2(dkernel_func(r[0]/grid_size)*wy,dkernel_func(r[1]/grid_size)*wx)/grid_size
                pic_v=pic_v+grid_vel[idx]*weight
                dvel=dvel+wp.outer(grid_vel[idx],dN)

    particle_vel_trial[t*num_particles+i]=pic_v
    particle_dvel[particle_next]=dvel

@wp.kernel
def update_pos(particle_pos:wp.array(dtype=wp.vec2),
               particle_vel:wp.array(dtype=wp.vec2),
               particle_vel_trial:wp.array(dtype=wp.vec2),
               t:int,
               num_particles:int,
               dt:float,
               ground:float,
               restitution:float,
               damping:float):
    i=wp.tid()
    current=t*num_particles+i
    next=(t+1)*num_particles+i
    vel=particle_vel_trial[t*num_particles+i]
    pos=particle_pos[current]+vel*dt
    if pos[1]<ground:
        pos=wp.vec2(pos[0],ground)
        if vel[1]<0.0:
            vel=wp.vec2(vel[0]*damping,-vel[1]*restitution)
    particle_pos[next]=pos
    particle_vel[next]=vel

@wp.kernel
def position_loss(particle_pos:wp.array(dtype=wp.vec2),
                  target_pos:wp.array(dtype=wp.vec2),
                  loss:wp.array(dtype=float),
                  t:int,
                  num_particles:int,
                  position_scale:float,
                  loss_scale:float):
    i=wp.tid()
    idx=(t+1)*num_particles+i
    error=(particle_pos[idx]-target_pos[idx])*position_scale
    wp.atomic_add(loss,0,wp.dot(error,error)*loss_scale)

@wp.kernel
def copy_particle_state(particle_pos:wp.array(dtype=wp.vec2),
                        particle_vel:wp.array(dtype=wp.vec2),
                        particle_dvel:wp.array(dtype=wp.mat22),
                        particle_F:wp.array(dtype=wp.mat22),
                        particle_P:wp.array(dtype=wp.mat22),
                        t:int,
                        num_particles:int):
    i=wp.tid()
    source=(t+1)*num_particles+i
    particle_pos[i]=particle_pos[source]
    particle_vel[i]=particle_vel[source]
    particle_dvel[i]=particle_dvel[source]
    particle_F[i]=particle_F[source]
    particle_P[i]=particle_P[source]
