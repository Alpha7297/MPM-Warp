import warp as wp

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

    def state_dict(self):
        return {
            "step_count":self.step_count,
            "beta1":self.beta1,
            "beta2":self.beta2,
            "epsilon":self.epsilon,
            "weight_decay":self.weight_decay,
            "first_moments":[moment.numpy().copy() for moment in self.first_moments],
            "second_moments":[moment.numpy().copy() for moment in self.second_moments],
        }

    def load_state_dict(self,state):
        first_moments=state["first_moments"]
        second_moments=state["second_moments"]
        if len(first_moments)!=len(self.parameters) or len(second_moments)!=len(self.parameters):
            raise ValueError("optimizer state parameter count does not match")
        for destination,values in zip(self.first_moments,first_moments):
            values=values.astype("float32",copy=False).reshape(-1)
            if values.size!=destination.size:
                raise ValueError("optimizer first moment shape does not match")
            destination.assign(values)
        for destination,values in zip(self.second_moments,second_moments):
            values=values.astype("float32",copy=False).reshape(-1)
            if values.size!=destination.size:
                raise ValueError("optimizer second moment shape does not match")
            destination.assign(values)
        self.step_count=int(state["step_count"])
        if self.step_count<0:
            raise ValueError("optimizer step_count must be non-negative")
        self.beta1=float(state["beta1"])
        self.beta2=float(state["beta2"])
        self.epsilon=float(state["epsilon"])
        self.weight_decay=float(state["weight_decay"])

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
