import os

import warp as wp

wp.config.kernel_cache_dir=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","outputs","warp_cache"))
wp.init()
device="cuda:0"

MODULUS=wp.constant(20000.0)
POISSON=wp.constant(0.2)
FLIP_RATIO=wp.constant(0.1)
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
        t=1.5-ar
        return 0.5*t*t
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
def init_grid(grid_pos:wp.array(dtype=wp.vec2),
              grid_vel:wp.array(dtype=wp.vec2),
              grid_vel_old:wp.array(dtype=wp.vec2),
              grid_f:wp.array(dtype=wp.vec2),
              grid_mass:wp.array(dtype=float),
              grid_size:float,
              grid_wid:int):
    i=wp.tid()
    x=i%grid_wid
    y=i//grid_wid
    grid_pos[i]=wp.vec2(float(x)*grid_size,float(y)*grid_size)
    grid_vel[i]=wp.vec2(0.0,0.0)
    grid_vel_old[i]=wp.vec2(0.0,0.0)
    grid_f[i]=wp.vec2(0.0,0.0)
    grid_mass[i]=0.0

@wp.kernel
def init_particle_state(particle_pos:wp.array(dtype=wp.vec2),
                        particle_vel:wp.array(dtype=wp.vec2),
                        particle_dvel:wp.array(dtype=wp.mat22),
                        particle_F:wp.array(dtype=wp.mat22),
                        particle_P:wp.array(dtype=wp.mat22),
                        particle_mass:wp.array(dtype=float),
                        grid_idx:wp.array(dtype=int),
                        m0:float,
                        grid_size:float,
                        grid_wid:int):
    i=wp.tid()
    pos=particle_pos[i]
    particle_vel[i]=wp.vec2(0.0,0.0)
    particle_dvel[i]=wp.mat22(0.0,0.0,0.0,0.0)
    particle_F[i]=wp.mat22(1.0,0.0,0.0,1.0)
    particle_P[i]=wp.mat22(0.0,0.0,0.0,0.0)
    particle_mass[i]=m0
    grid_idx[i]=int(pos[1]/grid_size)*grid_wid+int(pos[0]/grid_size)

@wp.kernel
def update_pos(particle_pos:wp.array(dtype=wp.vec2),
               particle_vel:wp.array(dtype=wp.vec2),
               grid_idx:wp.array(dtype=int),
               grid_size:float,
               grid_wid:int,
               dt:float
               ):
    i=wp.tid()
    pos=particle_pos[i]+particle_vel[i]*dt
    particle_pos[i]=pos
    grid_idx[i]=int(pos[1]/grid_size)*grid_wid+int(pos[0]/grid_size)

@wp.kernel
def theory_strain(particle_dvel:wp.array(dtype=wp.mat22),
                particle_F:wp.array(dtype=wp.mat22),
                particle_P:wp.array(dtype=wp.mat22),
                dt:float,
                mu:float,
                lbd:float
                ):
    i=wp.tid()
    F=particle_F[i]+(particle_dvel[i]@particle_F[i])*dt
    J=wp.determinant(F)
    LogJ=wp.log(J)
    particle_F[i]=F

    p00=mu*F[0,0]-F[1,1]*(mu-lbd*LogJ)/J
    p01=mu*F[0,1]+F[1,0]*(mu-lbd*LogJ)/J
    p10=mu*F[1,0]+F[0,1]*(mu-lbd*LogJ)/J
    p11=mu*F[1,1]-F[0,0]*(mu-lbd*LogJ)/J
    particle_P[i]=wp.mat22(p00,p01,p10,p11)

@wp.kernel
def update_F(particle_dvel:wp.array(dtype=wp.mat22),
             particle_F:wp.array(dtype=wp.mat22),
             dt:float):
    i=wp.tid()
    particle_F[i]=particle_F[i]+(particle_dvel[i]@particle_F[i])*dt

@wp.kernel
def zerolize_grids(grid_vel:wp.array(dtype=wp.vec2),
                   grid_f:wp.array(dtype=wp.vec2),
                   grid_mass:wp.array(dtype=float)):
    i=wp.tid()
    grid_vel[i]=wp.vec2(0.0,0.0)
    grid_f[i]=wp.vec2(0.0,0.0)
    grid_mass[i]=0.0

@wp.kernel
def P2G_update_grid(grid_pos:wp.array(dtype=wp.vec2),
                    grid_vel:wp.array(dtype=wp.vec2),
                    grid_f:wp.array(dtype=wp.vec2),
                    grid_mass:wp.array(dtype=float),
                    particle_pos:wp.array(dtype=wp.vec2),
                    particle_vel:wp.array(dtype=wp.vec2),
                    particle_mass:wp.array(dtype=float),
                    particle_F:wp.array(dtype=wp.mat22),
                    particle_P:wp.array(dtype=wp.mat22),
                    grid_size:float,
                    grid_wid:int,
                    grid_hei:int,
                    V0:float):
    i=wp.tid()
    pos=particle_pos[i]
    base_x=int(pos[0]/grid_size-0.5)
    base_y=int(pos[1]/grid_size-0.5)
    for ox in range(3):
        for oy in range(3):
            gx=base_x+ox
            gy=base_y+oy
            if gx>=0 and gx<grid_wid and gy>=0 and gy<grid_hei:
                idx=gy*grid_wid+gx
                r=pos-grid_pos[idx]
                wx=kernel_func(r[0]/grid_size)
                wy=kernel_func(r[1]/grid_size)
                weight=wx*wy
                wp.atomic_add(grid_vel,idx,particle_vel[i]*(particle_mass[i]*weight))
                wp.atomic_add(grid_mass,idx,particle_mass[i]*weight)

                dN=wp.vec2(dkernel_func(r[0]/grid_size)*wy,dkernel_func(r[1]/grid_size)*wx)
                PFt=particle_P[i]@wp.transpose(particle_F[i])
                wp.atomic_add(grid_f,idx,(PFt@dN)*(-V0/grid_size))

@wp.kernel
def P2G_grid_vel(grid_vel:wp.array(dtype=wp.vec2),
                 grid_mass:wp.array(dtype=float)):
    i=wp.tid()
    if grid_mass[i]>0.0:
        grid_vel[i]=grid_vel[i]/grid_mass[i]

@wp.kernel
def copy_vel(vel1:wp.array(dtype=wp.vec2),
             vel2:wp.array(dtype=wp.vec2)):
    i=wp.tid()
    vel2[i]=vel1[i]

@wp.kernel
def update_grid_vel(grid_vel:wp.array(dtype=wp.vec2),
                    grid_f:wp.array(dtype=wp.vec2),
                    grid_mass:wp.array(dtype=float),
                    f_extern:wp.vec2,
                    dt:float):
    i=wp.tid()
    if grid_mass[i]>0.0:
        grid_vel[i]=grid_vel[i]+dt*(grid_f[i]/grid_mass[i]+f_extern)

@wp.kernel
def G2P(grid_pos:wp.array(dtype=wp.vec2),
        grid_vel:wp.array(dtype=wp.vec2),
        grid_vel_old:wp.array(dtype=wp.vec2),
        particle_pos:wp.array(dtype=wp.vec2),
        particle_vel:wp.array(dtype=wp.vec2),
        particle_dvel:wp.array(dtype=wp.mat22),
        grid_size:float,
        grid_wid:int,
        grid_hei:int,
        flip_ratio:float):
    i=wp.tid()
    pos=particle_pos[i]
    old_v=particle_vel[i]
    pic_v=wp.vec2(0.0,0.0)
    flip_v=old_v
    dvel=wp.mat22(0.0,0.0,0.0,0.0)
    base_x=int(pos[0]/grid_size-0.5)
    base_y=int(pos[1]/grid_size-0.5)

    for ox in range(3):
        for oy in range(3):
            gx=base_x+ox
            gy=base_y+oy
            if gx>=0 and gx<grid_wid and gy>=0 and gy<grid_hei:
                idx=gy*grid_wid+gx
                r=pos-grid_pos[idx]
                wx=kernel_func(r[0]/grid_size)
                wy=kernel_func(r[1]/grid_size)
                weight=wx*wy
                dN=wp.vec2(dkernel_func(r[0]/grid_size)*wy,dkernel_func(r[1]/grid_size)*wx)/grid_size
                pic_v=pic_v+grid_vel[idx]*weight
                flip_v=flip_v+(grid_vel[idx]-grid_vel_old[idx])*weight
                dvel=dvel+wp.outer(grid_vel[idx],dN)

    particle_vel[i]=pic_v*(1.0-flip_ratio)+flip_v*flip_ratio
    particle_dvel[i]=dvel
