## MPM仿真理论

同样是混合拉格朗日与欧拉描述的方法

### 0.投影方法

离散情况下，质量动量分布为

$$
\rho(x)=\sum_p \delta(x-x_p) m_p\\
\rho(x)v(x)=\sum_p \delta(x-x_p) m_pv_p
$$

代入网格，取形函数$N(x)$

$$
m_i=\int \rho(x)N_i(x) d x=\sum_p N_i(x_p)m_p\\
m_iv_i=\sum_p N_i(x_p)m_pv_p
$$

能量

$$
E=\int \Psi(F) dX
$$

取变分

$$
\delta E=\int P:\delta F dX=\int P:\frac{\partial \delta x}{\partial x}\frac{\partial x}{\partial X}\frac{dx}{J}
$$

化简得到

$$
\delta E=\int \frac{1}{J}PF^T:\frac{\partial \delta x}{\partial x}dx=\int \sigma:\frac{\partial \delta x}{\partial x}dx
$$

从而得到

$$
\sigma=\frac{1}{J}PF^T
$$

投影到网格上

$$
f_i=\int N_i(x)\nabla\cdot \sigma dx=-\sum_p V_{p0}P_pF_p^T\nabla N_i(x_p)
$$

### 1.每个粒子储存信息

每个粒子储存初始位置，当前位置，质量，体积与形变梯度

之后用形变梯度计算应力

使用一阶Piola-Kirchhoff应力

$$
\Psi=\frac{\mu}{2}(\mathrm{tr}(F^TF)-2)-\mu \ln J+\frac{\lambda}{2}(\ln J)^2
$$

其中$J=\det(F)$

$$
P_{ij}=\frac{\partial \Psi}{\partial F_{ij}}
$$

$$
P_{11}=\mu a_{11}-\frac{a_{22}(\mu-\lambda \ln J)}{J}\\
P_{12}=\mu a_{12}-\frac{a_{21}(-\mu+\lambda \ln J)}{J}\\
P_{21}=\mu a_{21}-\frac{a_{12}(-\mu+\lambda \ln J)}{J}\\
P_{22}=\mu a_{22}-\frac{a_{11}(\mu-\lambda \ln J)}{J}
$$

### 2.P2G

清空网格信息，使用核函数将粒子信息投影到网格

在网格上用弹性力更新速度

### 3.G2P

更新粒子位置与形变梯度

由于MPM方法中最小单位不是三角形面，而是一个个粒子，因此形变梯度不会重新计算，更新方法是

$$
Fp_{n+1}=(I+\Delta t\cdot \nabla v_p)Fp_{n}\\
\nabla v_p=\sum_i v_i (\nabla N_i)^T
$$