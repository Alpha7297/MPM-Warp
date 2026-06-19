import os

import warp as wp

wp.config.kernel_cache_dir=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","outputs","warp_cache"))
wp.init()
device="cuda:0" if wp.is_cuda_available() else "cpu"

MODULUS=wp.constant(50000.0)
POISSON=wp.constant(0.2)
DENSITY=wp.constant(1000.0)
GRAVITY=wp.constant(9.8)
GROUND=wp.constant(0.134)
RESTITUTION=wp.constant(0.7)
DAMPING=wp.constant(0.9)

GRID_SIZE=wp.constant(0.02)
GRID_LEN=wp.constant(50)
GRID_HEI=wp.constant(50)
GRID_DEP=wp.constant(50)
NUM_GRIDS=wp.constant(GRID_LEN*GRID_HEI*GRID_DEP)

PARTICLES_PER_AXIS=wp.constant(2)
PARTICLES_PER_GRID=wp.constant(PARTICLES_PER_AXIS*PARTICLES_PER_AXIS*PARTICLES_PER_AXIS)
PARTICLE_SPACING=wp.constant(GRID_SIZE/float(PARTICLES_PER_AXIS))
V0=wp.constant(PARTICLE_SPACING*PARTICLE_SPACING*PARTICLE_SPACING)
M0=wp.constant(DENSITY*V0)

MU=wp.constant(MODULUS/(2.0*(1.0+POISSON)))
LAMBDA=wp.constant(MODULUS*POISSON/((1.0+POISSON)*(1.0-2.0*POISSON)))

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

@wp.func
def grid_index(gx:int,gy:int,gz:int,grid_wid:int,grid_hei:int):
    return gz*grid_wid*grid_hei+gy*grid_wid+gx

@wp.kernel
def init_grid(grid_pos:wp.array(dtype=wp.vec3),
              grid_vel:wp.array(dtype=wp.vec3),
              grid_f:wp.array(dtype=wp.vec3),
              grid_mass:wp.array(dtype=float),
              grid_size:float,
              grid_wid:int,
              grid_hei:int):
    i=wp.tid()
    x=i%grid_wid
    y=(i//grid_wid)%grid_hei
    z=i//(grid_wid*grid_hei)
    grid_pos[i]=wp.vec3(float(x)*grid_size,float(y)*grid_size,float(z)*grid_size)
    grid_vel[i]=wp.vec3(0.0,0.0,0.0)
    grid_f[i]=wp.vec3(0.0,0.0,0.0)
    grid_mass[i]=0.0

@wp.kernel
def init_particle_state(initial_pos:wp.array(dtype=wp.vec3),
                        initial_label:wp.array(dtype=int),
                        particle_pos:wp.array(dtype=wp.vec3),
                        particle_vel:wp.array(dtype=wp.vec3),
                        particle_dvel:wp.array(dtype=wp.mat33),
                        particle_F:wp.array(dtype=wp.mat33),
                        particle_P:wp.array(dtype=wp.mat33),
                        particle_mass:wp.array(dtype=float),
                        particle_label:wp.array(dtype=int),
                        m0:float):
    i=wp.tid()
    particle_pos[i]=initial_pos[i]
    particle_vel[i]=wp.vec3(0.0,0.0,0.0)
    particle_dvel[i]=wp.mat33(0.0,0.0,0.0,
                              0.0,0.0,0.0,
                              0.0,0.0,0.0)
    particle_F[i]=wp.mat33(1.0,0.0,0.0,
                           0.0,1.0,0.0,
                           0.0,0.0,1.0)
    particle_P[i]=wp.mat33(0.0,0.0,0.0,
                           0.0,0.0,0.0,
                           0.0,0.0,0.0)
    particle_mass[i]=m0
    particle_label[i]=initial_label[i]

@wp.kernel
def zerolize_grids(grid_vel:wp.array(dtype=wp.vec3),
                   grid_f:wp.array(dtype=wp.vec3),
                   grid_mass:wp.array(dtype=float)):
    i=wp.tid()
    grid_vel[i]=wp.vec3(0.0,0.0,0.0)
    grid_f[i]=wp.vec3(0.0,0.0,0.0)
    grid_mass[i]=0.0

@wp.kernel
def P2G_update_grid(grid_pos:wp.array(dtype=wp.vec3),
                    grid_vel:wp.array(dtype=wp.vec3),
                    grid_f:wp.array(dtype=wp.vec3),
                    grid_mass:wp.array(dtype=float),
                    particle_pos:wp.array(dtype=wp.vec3),
                    particle_vel:wp.array(dtype=wp.vec3),
                    particle_mass:wp.array(dtype=float),
                    particle_F:wp.array(dtype=wp.mat33),
                    particle_P:wp.array(dtype=wp.mat33),
                    grid_size:float,
                    grid_wid:int,
                    grid_hei:int,
                    grid_dep:int,
                    V0:float):
    i=wp.tid()
    pos=particle_pos[i]
    base_x=int(pos[0]/grid_size-0.5)
    base_y=int(pos[1]/grid_size-0.5)
    base_z=int(pos[2]/grid_size-0.5)
    for ox in range(3):
        for oy in range(3):
            for oz in range(3):
                gx=base_x+ox
                gy=base_y+oy
                gz=base_z+oz
                if gx>=0 and gx<grid_wid and gy>=0 and gy<grid_hei and gz>=0 and gz<grid_dep:
                    idx=grid_index(gx,gy,gz,grid_wid,grid_hei)
                    r=pos-grid_pos[idx]
                    wx=kernel_func(r[0]/grid_size)
                    wy=kernel_func(r[1]/grid_size)
                    wz=kernel_func(r[2]/grid_size)
                    weight=wx*wy*wz
                    wp.atomic_add(grid_vel,idx,particle_vel[i]*(particle_mass[i]*weight))
                    wp.atomic_add(grid_mass,idx,particle_mass[i]*weight)

                    dN=wp.vec3(dkernel_func(r[0]/grid_size)*wy*wz,
                               dkernel_func(r[1]/grid_size)*wx*wz,
                               dkernel_func(r[2]/grid_size)*wx*wy)
                    PFt=particle_P[i]@wp.transpose(particle_F[i])
                    wp.atomic_add(grid_f,idx,(PFt@dN)*(-V0/grid_size))

@wp.kernel
def P2G_grid_vel(grid_vel:wp.array(dtype=wp.vec3),
                 grid_mass:wp.array(dtype=float)):
    i=wp.tid()
    if grid_mass[i]>0.0:
        grid_vel[i]=grid_vel[i]/grid_mass[i]

@wp.kernel
def update_grid_vel(grid_vel:wp.array(dtype=wp.vec3),
                    grid_f:wp.array(dtype=wp.vec3),
                    grid_mass:wp.array(dtype=float),
                    f_extern:wp.vec3,
                    dt:float,
                    grid_size:float,
                    grid_wid:int,
                    grid_hei:int,
                    grid_dep:int,
                    boundary:int):
    i=wp.tid()
    if grid_mass[i]>0.0:
        vel=grid_vel[i]+dt*(grid_f[i]/grid_mass[i]+f_extern)
        x=i%grid_wid
        y=(i//grid_wid)%grid_hei
        z=i//(grid_wid*grid_hei)
        if x<boundary and vel[0]<0.0:
            vel=wp.vec3(0.0,vel[1],vel[2])
        if x>grid_wid-boundary-1 and vel[0]>0.0:
            vel=wp.vec3(0.0,vel[1],vel[2])
        if y<boundary and vel[1]<0.0:
            vel=wp.vec3(vel[0],0.0,vel[2])
        if z<boundary and vel[2]<0.0:
            vel=wp.vec3(vel[0],vel[1],0.0)
        if z>grid_dep-boundary-1 and vel[2]>0.0:
            vel=wp.vec3(vel[0],vel[1],0.0)
        grid_vel[i]=vel

@wp.kernel
def apply_grid_ground_contact(grid_vel:wp.array(dtype=wp.vec3),
                              grid_mass:wp.array(dtype=float),
                              grid_size:float,
                              grid_wid:int,
                              grid_hei:int,
                              ground:float,
                              restitution:float,
                              damping:float):
    i=wp.tid()
    y=(i//grid_wid)%grid_hei
    ground_y=int(ground/grid_size-0.5)
    if grid_mass[i]>0.0 and y>=ground_y and y<ground_y+3:
        vel=grid_vel[i]
        if vel[1]<0.0:
            grid_vel[i]=wp.vec3(vel[0]*damping,-vel[1]*restitution,vel[2]*damping)

@wp.kernel
def G2P(grid_pos:wp.array(dtype=wp.vec3),
        grid_vel:wp.array(dtype=wp.vec3),
        particle_pos:wp.array(dtype=wp.vec3),
        particle_vel:wp.array(dtype=wp.vec3),
        particle_dvel:wp.array(dtype=wp.mat33),
        grid_size:float,
        grid_wid:int,
        grid_hei:int,
        grid_dep:int):
    i=wp.tid()
    pos=particle_pos[i]
    pic_v=wp.vec3(0.0,0.0,0.0)
    dvel=wp.mat33(0.0,0.0,0.0,
                  0.0,0.0,0.0,
                  0.0,0.0,0.0)
    base_x=int(pos[0]/grid_size-0.5)
    base_y=int(pos[1]/grid_size-0.5)
    base_z=int(pos[2]/grid_size-0.5)

    for ox in range(3):
        for oy in range(3):
            for oz in range(3):
                gx=base_x+ox
                gy=base_y+oy
                gz=base_z+oz
                if gx>=0 and gx<grid_wid and gy>=0 and gy<grid_hei and gz>=0 and gz<grid_dep:
                    idx=grid_index(gx,gy,gz,grid_wid,grid_hei)
                    r=pos-grid_pos[idx]
                    wx=kernel_func(r[0]/grid_size)
                    wy=kernel_func(r[1]/grid_size)
                    wz=kernel_func(r[2]/grid_size)
                    weight=wx*wy*wz
                    dN=wp.vec3(dkernel_func(r[0]/grid_size)*wy*wz,
                               dkernel_func(r[1]/grid_size)*wx*wz,
                               dkernel_func(r[2]/grid_size)*wx*wy)/grid_size
                    pic_v=pic_v+grid_vel[idx]*weight
                    dvel=dvel+wp.outer(grid_vel[idx],dN)

    particle_vel[i]=pic_v
    particle_dvel[i]=dvel

@wp.kernel
def update_pos(particle_pos:wp.array(dtype=wp.vec3),
               particle_vel:wp.array(dtype=wp.vec3),
               particle_dvel:wp.array(dtype=wp.mat33),
               particle_F:wp.array(dtype=wp.mat33),
               particle_P:wp.array(dtype=wp.mat33),
               grid_size:float,
               grid_wid:int,
               grid_hei:int,
               grid_dep:int,
               dt:float,
               ground:float,
               restitution:float,
               damping:float,
               mu:float,
               lbd:float):
    i=wp.tid()
    pos=particle_pos[i]+particle_vel[i]*dt
    vel=particle_vel[i]
    max_x=float(grid_wid-2)*grid_size
    max_y=float(grid_hei-2)*grid_size
    max_z=float(grid_dep-2)*grid_size
    min_x=grid_size
    min_z=grid_size

    if pos[1]<ground:
        pos=wp.vec3(pos[0],ground,pos[2])
    if pos[0]<min_x:
        pos=wp.vec3(min_x,pos[1],pos[2])
        if vel[0]<0.0:
            vel=wp.vec3(-vel[0]*restitution,vel[1]*damping,vel[2]*damping)
    if pos[0]>max_x:
        pos=wp.vec3(max_x,pos[1],pos[2])
        if vel[0]>0.0:
            vel=wp.vec3(-vel[0]*restitution,vel[1]*damping,vel[2]*damping)
    if pos[1]>max_y:
        pos=wp.vec3(pos[0],max_y,pos[2])
        if vel[1]>0.0:
            vel=wp.vec3(vel[0]*damping,-vel[1]*restitution,vel[2]*damping)
    if pos[2]<min_z:
        pos=wp.vec3(pos[0],pos[1],min_z)
        if vel[2]<0.0:
            vel=wp.vec3(vel[0]*damping,vel[1]*damping,-vel[2]*restitution)
    if pos[2]>max_z:
        pos=wp.vec3(pos[0],pos[1],max_z)
        if vel[2]>0.0:
            vel=wp.vec3(vel[0]*damping,vel[1]*damping,-vel[2]*restitution)

    particle_pos[i]=pos
    particle_vel[i]=vel

    F=particle_F[i]+(particle_dvel[i]@particle_F[i])*dt
    J=wp.determinant(F)
    if J<0.2:
        J=0.2
    LogJ=wp.log(J)
    FinvT=wp.transpose(wp.inverse(F))
    particle_F[i]=F
    particle_P[i]=mu*F+(lbd*LogJ-mu)*FinvT
