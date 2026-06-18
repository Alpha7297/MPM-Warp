# MPMWarp

本项目包含两个实验目录：`basicMPM` 和 `NCLaw`。以下命令均应在项目根目录执行，并使用 `MPM-Diff` Conda 环境。

```bash
conda activate MPM-Diff
```

## basicMPM

`basicMPM` 是一个基础的二维 Material Point Method（MPM）实现，用于展示粒子与网格之间的 P2G、网格更新和 G2P 流程。

只运行仿真并生成 USD：

```bash
python basicMPM/generate.py
```

结果保存在 `outputs/mpm.usd`。

运行仿真并生成 MP4 可视化：

```bash
python basicMPM/plot.py
```

结果保存在 `outputs/mpm.usd` 和 `outputs/videos/mpm.mp4`。

## NCLaw

`NCLaw` 是 Neural Constitutive Laws 核心思想的二维复现实验。网络、MPM 仿真和优化器均使用 Warp 实现。网络根据形变梯度预测一阶 Piola-Kirchhoff 应力，训练目标是预测坐标与传统 Neo-Hookean 模型目标坐标之间的 L2 误差。

训练网络：

```bash
python NCLaw/train.py
```

训练配置位于 `NCLaw/train.py`，Warp 权重检查点以 `.npz` 格式保存在 `NCLaw/net/`。当前默认配置从第 0 个 epoch 训练至第 3000 个 epoch。

分别生成传统本构模型和 NCLaw 模型的 USD：

```bash
python NCLaw/generate.py
```

结果保存在 `outputs/nclaw/`。

将生成的 USD 渲染为左右对比视频：

```bash
python NCLaw/plot.py
```

结果保存在：

- `outputs/videos/cube_compare.mp4`
- `outputs/videos/table_compare.mp4`

完整运行生成与可视化流程：

```bash
python NCLaw/generate.py
python NCLaw/plot.py
```
