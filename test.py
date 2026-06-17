import os
import numpy as np
import warp as wp
import warp.render as render

wp.init()

os.makedirs("outputs",exist_ok=True)

device="cuda:0"

renderer=render.UsdRenderer("outputs/line.usd",up_axis="y",fps=60,scaling=1.0)

@wp.kernel
def forward(x:wp.array(dtype=wp.vec2),v:wp.array(dtype=wp.vec2),a:wp.array(dtype=wp.vec2),dt:wp.float32):
    i=wp.tid()
    v[i]=v[i]+a[i]*dt
    x[i]=x[i]+v[i]*dt

@wp.kernel
def force(x:wp.array(dtype=wp.vec2),v:wp.array(dtype=wp.vec2),a:wp.array(dtype=wp.vec2),k:wp.float32,m:wp.float32,l0:wp.float32,n:wp.int32):
    i=wp.tid()

    if i==0:
        d=x[i]-x[i+1]
        length=wp.length(d)
        if length>1.0e-6:
            r=d/length
            a[i]=-k*(length-l0)*r/m

    elif i==n-1:
        d=x[i]-x[i-1]
        length=wp.length(d)
        if length>1.0e-6:
            r=d/length
            a[i]=-k*(length-l0)*r/m

    else:
        d=x[i]-x[i-1]
        length=wp.length(d)
        ai=wp.vec2(0.0,0.0)

        if length>1.0e-6:
            r=d/length
            ai=ai-k*(length-l0)*r/m

        d=x[i]-x[i+1]
        length=wp.length(d)

        if length>1.0e-6:
            r=d/length
            ai=ai-k*(length-l0)*r/m

        a[i]=ai

@wp.kernel
def vec2_to_vec3(x:wp.array(dtype=wp.vec2),x3:wp.array(dtype=wp.vec3)):
    i=wp.tid()
    p=x[i]
    x3[i]=wp.vec3(p[0],p[1],0.0)

n=100

x_np=np.zeros((n,2),dtype=np.float32)
x_np[:,0]=np.arange(n,dtype=np.float32)
x_np[:,1]=np.zeros(n,dtype=np.float32)

v_np=np.zeros((n,2),dtype=np.float32)
v_np[n//2:,1]=np.ones(n//2,dtype=np.float32)
a_np=np.zeros((n,2),dtype=np.float32)

x=wp.array(x_np,dtype=wp.vec2,device=device)
v=wp.array(v_np,dtype=wp.vec2,device=device)
a=wp.array(a_np,dtype=wp.vec2,device=device)
x3=wp.zeros(n,dtype=wp.vec3,device=device)

steps=1000
fps=60.0

for step in range(steps):
    wp.launch(kernel=force,dim=n,inputs=[x,v,a,1.0,1.0,1.0,n],device=device)
    wp.launch(kernel=forward,dim=n,inputs=[x,v,a,0.01],device=device)

    wp.launch(kernel=vec2_to_vec3,dim=n,inputs=[x,x3],device=device)
    wp.synchronize()

    renderer.begin_frame(step/fps)
    renderer.render_points("particles",points=x3.numpy(),radius=0.005,colors=(0.2,0.5,1.0),as_spheres=True)
    renderer.end_frame()

renderer.save()