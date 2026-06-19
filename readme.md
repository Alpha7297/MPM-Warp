# MPMWarp

本项目包含三个主要目录：`2D`、`3D` 和 `NCLaw`。以下命令均应在项目根目录执行，并使用 `MPM-Diff` Conda 环境。

```bash
conda activate MPM-Diff
```

## 2D

`2D` 是二维 Material Point Method（MPM）基础实现，用于展示粒子到网格（P2G）、网格更新、网格到粒子（G2P）和地面接触流程。

只运行仿真并生成 USD：

```bash
python 2D/generate.py
```

默认输出：

- `outputs/mpm.usd`

运行仿真并生成 MP4 可视化：

```bash
python 2D/plot.py
```

默认输出：

- `outputs/mpm.usd`
- `outputs/videos/mpm.mp4`

## 3D

`3D` 是三维 MPM 实现，支持 `cube` 和 `table` 两种模型。仿真结果先保存为采样后的 `.npz` 帧数据，再由 `plot.py` 渲染为交互预览或视频。当前 3D 初始模型会绕 `z` 轴按 `INITIAL_ROTATION_Z_DEGREES` 做顺时针旋转，常量位于 `3D/generate.py`。

生成默认 `cube` 帧数据：

```bash
python 3D/generate.py
```

默认输出：

- `outputs/3d_mpm/cube_frames.npz`

生成 `table` 帧数据：

```bash
python 3D/generate.py --model table
```

默认输出：

- `outputs/3d_mpm/table_frames.npz`

显示交互预览：

```bash
python 3D/plot.py --render show --model cube
```

生成 MP4 视频：

```bash
python 3D/plot.py --render video --model cube
python 3D/plot.py --render video --model table
```

默认输出：

- `outputs/3d_mpm/cube_video.mp4`
- `outputs/3d_mpm/table_video.mp4`

`3D/plot.py` 生成视频时会逐帧渲染到 `/tmp` 下的临时目录，并在写入视频后删除临时帧，以降低内存占用。

## NCLaw

`NCLaw` 是 Neural Constitutive Laws 的二维复现实验。网络、MPM 仿真和优化器均使用 Warp 实现。网络根据形变梯度预测一阶 Piola-Kirchhoff 应力，训练目标是预测坐标与传统 Neo-Hookean 目标坐标之间的 L2 误差。

训练网络：

```bash
python NCLaw/train.py
```

训练配置位于 `NCLaw/train.py`。Warp 权重检查点以 `.npz` 格式保存在 `NCLaw/net/`。当前默认配置从第 0 个 epoch 训练至第 3000 个 epoch。

分别生成传统本构模型和 NCLaw 模型的 USD：

```bash
python NCLaw/generate.py
```

默认输出：

- `outputs/nclaw/cube_tradition.usd`
- `outputs/nclaw/cube_nclaw.usd`
- `outputs/nclaw/table_tradition.usd`
- `outputs/nclaw/table_nclaw.usd`

将生成的 USD 渲染为左右对比视频：

```bash
python NCLaw/plot.py
```

默认输出：

- `outputs/videos/cube_compare.mp4`
- `outputs/videos/table_compare.mp4`

完整运行生成与可视化流程：

```bash
python NCLaw/generate.py
python NCLaw/plot.py
```
