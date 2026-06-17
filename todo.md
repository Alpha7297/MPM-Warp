# Differentiable 2D MPM With NVIDIA Warp:One-Week TODO

基准日期:2026-06-17

目标:一周内做出一个最小可用的 2D 可微分 MPM 模拟器。先实现稳定的显式 2D MPM 前向仿真，再用 NVIDIA Warp 的 kernel、array、Tape 和自动微分能力验证端到端梯度。

参考入口:

- NVIDIA Warp 文档:https://nvidia.github.io/warp/stable/
- Warp GitHub:https://github.com/NVIDIA/warp
- Warp Basics:https://nvidia.github.io/warp/stable/user_guide/basics.html
- Warp Differentiability:https://nvidia.github.io/warp/stable/user_guide/differentiability.html
- Warp Interoperability:https://nvidia.github.io/warp/stable/user_guide/interoperability.html

## Day 1:Warp 环境和最小 kernel

- [ ] 安装环境:Python>=3.10，CUDA 可用时安装 `warp-lang`，需要示例时安装 `warp-lang[examples]`。
- [ ] 运行一次 `wp.init()`，记录 CPU/CUDA 设备、驱动、Warp 版本和 kernel cache 路径。
- [ ] 写一个最小 Warp kernel:输入 `wp.array[wp.vec2]`，更新粒子位置，分别在 `cpu` 和 `cuda:0` 上运行。
- [ ] 学习并记录 Warp 基础规则:kernel 必须静态类型标注；用 `@wp.kernel` 和 `wp.launch`；kernel 内用 `wp.tid()` 取线程 id；数组用 `wp.array`、`wp.array2d`；kernel 内避免普通 Python list。
- [ ] 确认 `wp.vec2`、`wp.mat22`、`wp.atomic_add`、`wp.zeros`、`wp.empty`、`array.numpy()` 的基本用法。
- [ ] 产出:一个 `examples/warp_sanity.py`，能在 CPU 或 GPU 上完成一次粒子平移。
- [ ] 检查:10 个粒子平移一步后，CPU 读回结果与 NumPy 预期一致。

## Day 2:先写 NumPy 版 2D MPM 前向基线

- [ ] 固定最小算法:显式 MPM、2D、规则网格、二次 B-spline 权重、APIC/MLS-MPM 风格 P2G/G2P。
- [ ] 定义粒子状态:`x`、`v`、`C`、`F`、`mass`、`volume`。第一版可只用弹性材料，不先做塑性。
- [ ] 定义网格状态:`grid_m`、`grid_v`。每步执行 `clear_grid`、`p2g`、`grid_update`、`g2p`。
- [ ] 在 NumPy 中实现 3x3 stencil 权重，并验证权重和接近 1。
- [ ] 加入简单边界条件:靠近边界时钳制或反射网格速度。
- [ ] 用小规模参数开始:32x32 网格，几百个粒子，短时间步。
- [ ] 产出:`mpm2d_numpy.py`。先追求清晰和正确，不追求性能。
- [ ] 检查:无外力时质心基本保持；有重力时粒子整体下落；每步总质量守恒。

## Day 3:把 2D MPM 前向仿真移植到 Warp

- [ ] 规划 Warp 数组布局:粒子数组用一维 `wp.array`；网格质量用 `wp.array2d[float]`；网格速度用 `wp.array2d[wp.vec2]`。
- [ ] 写 kernel:`clear_grid_kernel`、`p2g_kernel`、`grid_update_kernel`、`g2p_kernel`。
- [ ] 在 `p2g_kernel` 中用 `wp.atomic_add` 累加 `grid_m` 和 `grid_v`，避免粒子并行写网格的竞争。
- [ ] 在 `g2p_kernel` 中从 3x3 网格节点插值回粒子，更新 `v`、`C`、`F`、`x`。
- [ ] 先实现可运行的线性弹性或简化 Neo-Hookean 应力，再扩展到固定 corotated。
- [ ] 把所有可调常量集中管理:`dx`、`inv_dx`、`dt`、`gravity`、`bound`、`mu`、`lambda`。
- [ ] 产出:`mpm2d_warp.py`。接口至少包含 `step()` 和 `simulate(num_steps)`。
- [ ] 检查:Warp 版在小规模场景下与 NumPy 版趋势一致；总质量守恒；无 NaN/Inf。

## Day 4:前向稳定性和可视化

- [ ] 加入简单初始化:方块、圆盘、多个材料点云。
- [ ] 加入基础可视化:用 Matplotlib 保存粒子散点图或动画帧。
- [ ] 扫描稳定参数:减小 `dt`、调整 `mu` 和 `lambda`，记录不会爆炸的默认值。
- [ ] 加入基本诊断:总质量、质心、速度最大值、粒子是否越界、NaN/Inf 检测。
- [ ] 对 Warp kernel 做一次简单 profiling:记录首次编译时间和后续运行时间，区分 CPU/GPU。
- [ ] 产出:`scripts/run_forward.py` 和 `outputs/forward_smoke/`。
- [ ] 检查:至少跑 100 步不爆炸，并能保存最后一帧粒子图。

## Day 5:接入 Warp 自动微分

- [ ] 选择第一版可微目标:优化初始速度 `v0`，让最终质心或少量粒子位置接近目标。
- [ ] 给需要求导的数组设置 `requires_grad=True`，用 `wp.Tape()` 包住多步仿真和 loss kernel。
- [ ] 写 `loss_kernel`:输入最终粒子位置和目标位置，输出标量 loss。
- [ ] 避免覆盖参与计算图的数组。对每个时间步使用 ping-pong buffer 或保存必要中间状态。
- [ ] 打开 `wp.config.verify_autograd_array_access=True` 检查常见数组覆盖错误。
- [ ] 调用 `tape.backward(loss)` 或用 `tape.backward(grads={output:seed})` 验证梯度。
- [ ] 做有限差分检查:随机选 3 到 5 个 `v0` 分量，对比 Warp 梯度和 finite difference。
- [ ] 产出:`examples/diff_mpm_smoke.py`。
- [ ] 检查:loss 对 `v0` 的梯度非零，有限差分相对误差在可接受范围内。

## Day 6:做一个小型反问题或优化 demo

- [ ] 实现梯度下降或 Adam。第一版可直接在 Warp 数组和 NumPy 之间同步，也可用 PyTorch interop 做优化器。
- [ ] 优化变量优先级:`v0` 最简单；其次是重力、材料参数 `mu/lambda`；最后才考虑粒子初始位置或控制力。
- [ ] 固定短 horizon:先 10 到 30 步，确认梯度稳定后再加长。
- [ ] 记录每轮 loss、梯度范数、参数范围和最终图像。
- [ ] 如果内存压力大，先减少粒子数和步数，再考虑 checkpoint 或重放策略。
- [ ] 产出:`scripts/optimize_initial_velocity.py`。
- [ ] 检查:20 到 100 轮优化内 loss 明显下降，最终状态比初始状态更接近目标。

## Day 7:整理接口、测试和下一步路线

- [ ] 整理项目结构:核心仿真、示例脚本、测试、输出目录分开。
- [ ] 写最小测试:权重和为 1；质量守恒；单步无 NaN；loss backward 后梯度存在。
- [ ] 写 README:安装、运行前向仿真、运行可微优化、已知限制。
- [ ] 标注当前限制:只支持 2D；显式积分；简单边界；材料模型有限；长 horizon 梯度可能不稳定。
- [ ] 决定下一周扩展方向:更好的材料模型、塑性、碰撞、稀疏网格、checkpoint、PyTorch/JAX 训练管线。
- [ ] 产出:`README.md`、`tests/test_mpm2d.py`、一张前向图、一张优化前后对比图。
- [ ] 检查:从干净环境按 README 能跑通 forward smoke 和 diff smoke。

## 一周结束时的验收标准

- [ ] 有一个可运行的 2D MPM 前向模拟。
- [ ] 有一个 Warp kernel 版本，不只是 NumPy 原型。
- [ ] 有一个通过 `wp.Tape()` 反传的可微 demo。
- [ ] 有一个有限差分梯度检查。
- [ ] 有一个小型优化例子，loss 能下降。
- [ ] 有最小文档和最小测试。

## 实现顺序原则

- [ ] 先 CPU/NumPy 正确，再 Warp 性能。
- [ ] 先前向稳定，再自动微分。
- [ ] 先优化 `v0`，再优化材料参数。
- [ ] 先短 horizon 和少粒子，再扩大规模。
- [ ] 每天结束都保留一个能跑通的版本。
