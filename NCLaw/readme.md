## NCLaw

NCLaw复现小实验，使用轨迹拟合应力应变关系

仿真器的应力应变关系为

```python
    J=wp.determinant(F)
    LogJ=wp.log(J)
    particle_F[i]=F

    p00=mu*F[0,0]-F[1,1]*(mu-lbd*LogJ)/J
    p01=mu*F[0,1]+F[1,0]*(mu-lbd*LogJ)/J
    p10=mu*F[1,0]+F[0,1]*(mu-lbd*LogJ)/J
    p11=mu*F[1,1]-F[0,0]*(mu-lbd*LogJ)/J
    particle_P[i]=wp.mat22(p00,p01,p10,p11)
```

神经网络结构为

7->64->64->4

输入$\Sigma$奇异值，$\F^T F$与$\det{F}$

输出应力$P$

激活函数使用GeLU，loss使用位置的L2误差