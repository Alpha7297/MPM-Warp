import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self,hidden_dim=64):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(7,hidden_dim,bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim,hidden_dim,bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim,4,bias=False),
        )

    def features(self,F):
        S=torch.linalg.svdvals(F)-1.0
        C=F.transpose(-1,-2)@F
        C0=torch.zeros_like(C)
        C0[:,0,0]=1.0
        C0[:,1,1]=1.0
        C=(C-C0).reshape(-1,4)
        J=torch.det(F).unsqueeze(-1)-1.0
        return torch.cat([S,C,J],dim=-1)

    def forward(self,F):
        U,_,Vh=torch.linalg.svd(F)
        R=U@Vh
        T1=self.net(self.features(F)).reshape(-1,2,2)
        T2=0.5*(T1+T1.transpose(-1,-2))
        return R@T2
