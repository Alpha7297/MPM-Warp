import warp as wp
import numpy as np

wp.init()
np.random.seed(13)
device="cpu"

@wp.func
def target_func(r:float):
    return wp.sin(r)

def target(r:float):
    return np.sin(r)

@wp.kernel
def relu(input:wp.array(dtype=float),
         output:wp.array(dtype=float)):
    i=wp.tid()
    output[i]=wp.max(input[i],0.0)

@wp.kernel
def multiply(input:wp.array(dtype=float),
             weight:wp.array(dtype=float),
             bias:wp.array(dtype=float),
             output:wp.array(dtype=float),
             indim:int,
             outdim:int):
    i=wp.tid()
    value=bias[i]
    for j in range(indim):
        value=value+weight[i*indim+j]*input[j]
    output[i]=value

@wp.kernel
def l2_loss(output:wp.array(dtype=float),
            _data3:wp.array(dtype=float),
            loss:wp.array(dtype=float)):
    i=wp.tid()
    diff=_data3[i]-output[i]
    wp.atomic_add(loss,0,diff*diff)

@wp.kernel
def sgd(param:wp.array(dtype=float),
        grad:wp.array(dtype=float),
        lr:float):
    i=wp.tid()
    param[i]=param[i]-lr*grad[i]

train_x=np.linspace(0.0,2.0*np.pi,256,dtype=np.float32)
train_y=target(train_x).astype(np.float32)

test_x=(np.random.rand(10).astype(np.float32)*2.0*np.pi)
test_y=target(test_x).astype(np.float32)

#1->16->8->1

linear1=wp.array((np.random.randn(16)*0.1).astype(np.float32),dtype=float,device=device,requires_grad=True)
bias1=wp.array(np.zeros(16,dtype=np.float32),dtype=float,device=device,requires_grad=True)
linear2=wp.array((np.random.randn(8*16)*0.1).astype(np.float32),dtype=float,device=device,requires_grad=True)
bias2=wp.array(np.zeros(8,dtype=np.float32),dtype=float,device=device,requires_grad=True)
linear3=wp.array((np.random.randn(8)*0.1).astype(np.float32),dtype=float,device=device,requires_grad=True)
bias3=wp.array(np.zeros(1,dtype=np.float32),dtype=float,device=device,requires_grad=True)

params=[linear1,bias1,linear2,bias2,linear3,bias3]

def scalar_array(value):
    if isinstance(value,wp.array):
        return value
    return wp.array([float(value)],dtype=float,device=device)

def zero_grads():
    for param in params:
        param.grad.zero_()

def sgd_step(lr):
    for param in params:
        wp.launch(
            kernel=sgd,
            dim=param.shape[0],
            inputs=[param,param.grad,lr],
            device=device,
        )

def forward_layers(input,requires_grad=False):
    _data1=wp.zeros(16,dtype=float,device=device,requires_grad=requires_grad)
    wp.launch(
        kernel=multiply,
        dim=16,
        inputs=[input,linear1,bias1,_data1,1,16],
        device=device,
    )
    _reludata1=wp.zeros(16,dtype=float,device=device,requires_grad=requires_grad)
    wp.launch(
        kernel=relu,
        dim=16,
        inputs=[_data1,_reludata1],
        device=device,
    )
    _data2=wp.zeros(8,dtype=float,device=device,requires_grad=requires_grad)
    wp.launch(
        kernel=multiply,
        dim=8,
        inputs=[_reludata1,linear2,bias2,_data2,16,8],
        device=device,
    )
    _reludata2=wp.zeros(8,dtype=float,device=device,requires_grad=requires_grad)
    wp.launch(
        kernel=relu,
        dim=8,
        inputs=[_data2,_reludata2],
        device=device,
    )
    _data3=wp.zeros(1,dtype=float,device=device,requires_grad=requires_grad)
    wp.launch(
        kernel=multiply,
        dim=1,
        inputs=[_reludata2,linear3,bias3,_data3,8,1],
        device=device,
    )
    return _data3

def forward(input,output=None,eval=True):
    input=scalar_array(input)
    if eval:
        _data3=forward_layers(input,requires_grad=False)
        wp.synchronize()
        return float(_data3.numpy()[0])

    if output is None:
        raise ValueError("output is required when eval=False")

    output=scalar_array(output)
    loss=wp.zeros(1,dtype=float,device=device,requires_grad=True)
    zero_grads()
    with wp.Tape() as tape:
        _data3=forward_layers(input,requires_grad=True)
        wp.launch(
            kernel=l2_loss,
            dim=1,
            inputs=[output,_data3,loss],
            device=device,
        )
    tape.backward(loss)
    wp.synchronize()
    return float(loss.numpy()[0])

def train_batch(batch_indices):
    loss=wp.zeros(1,dtype=float,device=device,requires_grad=True)
    zero_grads()
    with wp.Tape() as tape:
        for idx in batch_indices:
            input=scalar_array(train_x[idx])
            output=scalar_array(train_y[idx])
            _data3=forward_layers(input,requires_grad=True)
            wp.launch(
                kernel=l2_loss,
                dim=1,
                inputs=[output,_data3,loss],
                device=device,
            )
    tape.backward(loss)
    wp.synchronize()
    batch_len=float(len(batch_indices))
    return float(loss.numpy()[0])/batch_len

batch_size=1
train_epoch=500

def train():
    losses=[]
    indices=np.arange(train_x.shape[0])
    for epoch in range(train_epoch):
        np.random.shuffle(indices)
        epoch_loss=0.0
        for start in range(0,train_x.shape[0],batch_size):
            batch_indices=indices[start:start+batch_size]
            loss=train_batch(batch_indices)
            learning_rate=1e-3
            sgd_step(learning_rate/float(len(batch_indices)))
            epoch_loss=epoch_loss+loss*float(len(batch_indices))
        epoch_loss=epoch_loss/float(train_x.shape[0])
        losses.append(epoch_loss)
        print("epoch",epoch,"loss",epoch_loss)
    return losses

def predict(xs):
    return np.array([forward(x,eval=True) for x in xs],dtype=np.float32)

if __name__=="__main__":
    train()
    print("test_y",test_y)
    print("pred_y",predict(test_x))
