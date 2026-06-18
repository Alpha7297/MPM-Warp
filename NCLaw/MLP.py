import math
import os

import numpy as np
import warp as wp

wp.config.kernel_cache_dir=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","outputs","warp_cache"))

INPUT_DIM=7
HIDDEN_DIM=64
OUTPUT_DIM=4

@wp.kernel
def build_features(particle_F:wp.array(dtype=wp.mat22),
                   features:wp.array(dtype=float),
                   rotations:wp.array(dtype=wp.mat22),
                   t:int,
                   batch_size:int):
    i=wp.tid()
    particle_offset=(t+1)*batch_size
    feature_offset=t*batch_size*INPUT_DIM
    F=particle_F[particle_offset+i]
    U,sigma,V=wp.svd2(F)
    R=U@wp.transpose(V)
    C=wp.transpose(F)@F
    offset=feature_offset+i*INPUT_DIM
    features[offset]=sigma[0]-1.0
    features[offset+1]=sigma[1]-1.0
    features[offset+2]=C[0,0]-1.0
    features[offset+3]=C[0,1]
    features[offset+4]=C[1,0]
    features[offset+5]=C[1,1]-1.0
    features[offset+6]=wp.determinant(F)-1.0
    rotations[t*batch_size+i]=R

@wp.kernel
def matrix_multiply(x:wp.array(dtype=float),
                    weight:wp.array(dtype=float),
                    output:wp.array(dtype=float),
                    t:int,
                    batch_size:int,
                    in_dim:int,
                    out_dim:int):
    tid=wp.tid()
    batch=tid//out_dim
    col=tid%out_dim
    x_offset=t*batch_size*in_dim+batch*in_dim
    output_offset=t*batch_size*out_dim+batch*out_dim
    value=float(0.0)
    for k in range(in_dim):
        value=value+x[x_offset+k]*weight[k*out_dim+col]
    output[output_offset+col]=value

@wp.kernel
def gelu(x:wp.array(dtype=float),
         output:wp.array(dtype=float),
         t:int,
         batch_size:int,
         dim:int):
    tid=wp.tid()
    offset=t*batch_size*dim+tid
    value=x[offset]
    output[offset]=0.5*value*(1.0+wp.erf(value*0.7071067811865476))

@wp.kernel
def build_stress(raw_output:wp.array(dtype=float),
                 rotations:wp.array(dtype=wp.mat22),
                 particle_P:wp.array(dtype=wp.mat22),
                 t:int,
                 batch_size:int):
    i=wp.tid()
    output_offset=t*batch_size*OUTPUT_DIM+i*OUTPUT_DIM
    T1=wp.mat22(
        raw_output[output_offset],
        raw_output[output_offset+1],
        raw_output[output_offset+2],
        raw_output[output_offset+3],
    )
    T2=0.5*(T1+wp.transpose(T1))
    particle_P[(t+1)*batch_size+i]=rotations[t*batch_size+i]@T2

@wp.kernel
def adamw_step(parameter:wp.array(dtype=float),
               gradient:wp.array(dtype=float),
               first_moment:wp.array(dtype=float),
               second_moment:wp.array(dtype=float),
               lr:float,
               beta1:float,
               beta2:float,
               beta1_power:float,
               beta2_power:float,
               epsilon:float,
               weight_decay:float):
    i=wp.tid()
    grad=gradient[i]
    moment1=beta1*first_moment[i]+(1.0-beta1)*grad
    moment2=beta2*second_moment[i]+(1.0-beta2)*grad*grad
    first_moment[i]=moment1
    second_moment[i]=moment2
    corrected1=moment1/(1.0-beta1_power)
    corrected2=moment2/(1.0-beta2_power)
    decay=lr*weight_decay*parameter[i]
    parameter[i]=parameter[i]-decay-lr*corrected1/(wp.sqrt(corrected2)+epsilon)

class MLP:
    def __init__(self,batch_size,max_steps,device,hidden_dim=HIDDEN_DIM,seed=0):
        self.batch_size=batch_size
        self.max_steps=max_steps
        self.device=device
        self.hidden_dim=hidden_dim
        rng=np.random.default_rng(seed)
        self.w1=self._weight(INPUT_DIM,hidden_dim,rng)
        self.w2=self._weight(hidden_dim,hidden_dim,rng)
        self.w3=self._weight(hidden_dim,OUTPUT_DIM,rng)
        self.parameters=[self.w1,self.w2,self.w3]
        self.features=wp.zeros(max_steps*batch_size*INPUT_DIM,dtype=float,device=device,requires_grad=True)
        self.linear1=wp.zeros(max_steps*batch_size*hidden_dim,dtype=float,device=device,requires_grad=True)
        self.hidden1=wp.zeros(max_steps*batch_size*hidden_dim,dtype=float,device=device,requires_grad=True)
        self.linear2=wp.zeros(max_steps*batch_size*hidden_dim,dtype=float,device=device,requires_grad=True)
        self.hidden2=wp.zeros(max_steps*batch_size*hidden_dim,dtype=float,device=device,requires_grad=True)
        self.raw_output=wp.zeros(max_steps*batch_size*OUTPUT_DIM,dtype=float,device=device,requires_grad=True)
        self.rotations=wp.zeros(max_steps*batch_size,dtype=wp.mat22,device=device,requires_grad=True)
        self.workspace=[
            self.features,
            self.linear1,
            self.hidden1,
            self.linear2,
            self.hidden2,
            self.raw_output,
            self.rotations,
        ]

    def _weight(self,in_dim,out_dim,rng):
        limit=math.sqrt(6.0/float(in_dim+out_dim))
        values=rng.uniform(-limit,limit,size=in_dim*out_dim).astype(np.float32)
        return wp.array(values,dtype=float,device=self.device,requires_grad=True)

    def forward(self,particle_F,particle_P,t):
        wp.launch(
            build_features,
            dim=self.batch_size,
            inputs=[particle_F,self.features,self.rotations,t,self.batch_size],
            device=self.device,
        )
        wp.launch(
            matrix_multiply,
            dim=self.batch_size*self.hidden_dim,
            inputs=[self.features,self.w1,self.linear1,t,self.batch_size,INPUT_DIM,self.hidden_dim],
            device=self.device,
        )
        wp.launch(
            gelu,
            dim=self.batch_size*self.hidden_dim,
            inputs=[self.linear1,self.hidden1,t,self.batch_size,self.hidden_dim],
            device=self.device,
        )
        wp.launch(
            matrix_multiply,
            dim=self.batch_size*self.hidden_dim,
            inputs=[self.hidden1,self.w2,self.linear2,t,self.batch_size,self.hidden_dim,self.hidden_dim],
            device=self.device,
        )
        wp.launch(
            gelu,
            dim=self.batch_size*self.hidden_dim,
            inputs=[self.linear2,self.hidden2,t,self.batch_size,self.hidden_dim],
            device=self.device,
        )
        wp.launch(
            matrix_multiply,
            dim=self.batch_size*OUTPUT_DIM,
            inputs=[self.hidden2,self.w3,self.raw_output,t,self.batch_size,self.hidden_dim,OUTPUT_DIM],
            device=self.device,
        )
        wp.launch(
            build_stress,
            dim=self.batch_size,
            inputs=[self.raw_output,self.rotations,particle_P,t,self.batch_size],
            device=self.device,
        )

    def zero_workspace(self):
        for array in self.workspace:
            array.zero_()
            array.grad.zero_()

    def save(self,path,epoch,loss):
        np.savez(
            path,
            epoch=np.int64(epoch),
            loss=np.float32(loss),
            w1=self.w1.numpy(),
            w2=self.w2.numpy(),
            w3=self.w3.numpy(),
        )

    def load(self,path):
        checkpoint=np.load(path)
        self.w1.assign(checkpoint["w1"])
        self.w2.assign(checkpoint["w2"])
        self.w3.assign(checkpoint["w3"])
        return int(checkpoint["epoch"]),float(checkpoint["loss"])

class AdamW:
    def __init__(self,parameters,beta1=0.9,beta2=0.999,epsilon=1.0e-8,weight_decay=1.0e-2):
        self.parameters=parameters
        self.beta1=beta1
        self.beta2=beta2
        self.epsilon=epsilon
        self.weight_decay=weight_decay
        self.step_count=0
        self.first_moments=[wp.zeros_like(parameter,requires_grad=False) for parameter in parameters]
        self.second_moments=[wp.zeros_like(parameter,requires_grad=False) for parameter in parameters]

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.grad.zero_()

    def step(self,lr):
        self.step_count+=1
        beta1_power=self.beta1**self.step_count
        beta2_power=self.beta2**self.step_count
        for parameter,first_moment,second_moment in zip(self.parameters,self.first_moments,self.second_moments):
            wp.launch(
                adamw_step,
                dim=parameter.size,
                inputs=[
                    parameter,
                    parameter.grad,
                    first_moment,
                    second_moment,
                    lr,
                    self.beta1,
                    self.beta2,
                    beta1_power,
                    beta2_power,
                    self.epsilon,
                    self.weight_decay,
                ],
                device=parameter.device,
            )
