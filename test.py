import numpy as np
import warp as wp

wp.init()

@wp.kernel
def translate_kernel(x:wp.array(dtype=wp.vec2),v:wp.array(dtype=wp.vec2),dt:float):
    i=int(wp.tid())
    x[i]=x[i]+v[i]*dt

device="cuda:0"

n=10
x_np=np.zeros((n,2),dtype=float)
v_np=np.ones((n,2),dtype=float)

x=wp.array(x_np,dtype=wp.vec2,device=device)
v=wp.array(v_np,dtype=wp.vec2,device=device)

wp.launch(
    kernel=translate_kernel,
    dim=n,
    inputs=[x,v,0.01],
    device=device,
)

print(x.numpy())