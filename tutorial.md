# NVIDIA Warp DSL 入门教程

本文面向这个项目的目标:用 NVIDIA Warp 写一个最小 2D 可微分 MPM。结论先说:Warp 不是一门完全独立的新语言，而是嵌在 Python 里的并行计算 DSL。你用 Python 分配数组、组织仿真流程、发起 kernel；用 `@wp.kernel` 标记的函数会被 Warp JIT 编译成 CPU/CUDA 原生代码。

官方文档当前稳定版是 Warp 1.14.0。本文只覆盖项目早期最需要的部分:基础函数、decorator、并行模型、tile API、自动微分和 MPM kernel 拆分方式。

## 1. 最小使用方式

安装包名是 `warp-lang`，Python 代码里导入名是 `warp`:

```bash
python -m pip install warp-lang
```

如果需要官方 examples:

```bash
python -m pip install "warp-lang[examples]"
```

最小 kernel:

```python
import numpy as np
import warp as wp

wp.init()

@wp.kernel
def translate_kernel(x:wp.array(dtype=wp.vec2),v:wp.array(dtype=wp.vec2),dt:float):
    i=wp.tid()
    x[i]=x[i]+v[i]*dt

device="cpu"
# 如果有 CUDA,可以改成 "cuda:0"

n=10
x_np=np.zeros((n,2),dtype=np.float32)
v_np=np.ones((n,2),dtype=np.float32)

x=wp.array(x_np,dtype=wp.vec2,device=device)
v=wp.array(v_np,dtype=wp.vec2,device=device)

wp.launch(
    kernel=translate_kernel,
    dim=n,
    inputs=[x,v,0.01],
    device=device,
)

print(x.numpy())
```

这里的关键点:

- `wp.init()` 初始化 Warp runtime。
- `@wp.kernel` 声明一个可编译 kernel。
- `wp.launch(...,dim=n,...)` 启动 `n` 个逻辑线程。
- `wp.tid()` 得到当前线程编号。
- kernel 参数必须有静态类型标注，数组参数写成 `x:wp.array(dtype=wp.vec2)`。
- kernel 不返回值，结果通常写回传入的 `wp.array`。

## 2. Python Scope 和 Kernel Scope

Warp 代码有两个层级:

- Python scope:普通 Python。负责创建数组、加载数据、调用 `wp.launch()`、保存图片、做训练循环。
- Kernel scope:`@wp.kernel` 和 `@wp.func` 里的代码。会被 Warp 编译，语法像 Python，但不是完整 Python。

Kernel scope 常见限制:

- 参数必须类型标注。
- 不能随便访问 Python 全局对象。
- 不要在 kernel 里用普通 Python list。
- 小的固定长度数据用 `wp.vec2`、`wp.vec3`、`wp.mat22`、`wp.zeros(shape=...,dtype=...)`。
- tuple 初始化经常不适合作为 kernel 内数据结构，优先显式构造 Warp 类型。
- 分支、循环、数学表达式可以用，但要让类型清楚。

错误示例:

```python
@wp.kernel
def bad_kernel(out:wp.array(dtype=float)):
    a=[1.0,2.0,3.0]
    out[0]=a[0]
```

更合适:

```python
@wp.kernel
def good_kernel(out:wp.array(dtype=float)):
    a=wp.vec3(1.0,2.0,3.0)
    out[0]=a[0]
```

## 3. 常用数据类型

标量:

- `int`
- `float`
- `bool`
- `wp.int32`、`wp.float32` 等更明确的类型

向量:

- `wp.vec2`
- `wp.vec3`
- `wp.vec4`
- 整型向量如 `wp.vec2i`

矩阵:

- `wp.mat22`
- `wp.mat33`
- `wp.mat44`

数组参数注释使用调用式写法，不使用 `wp.array[...]`:

- `wp.array(dtype=T)`
- `wp.array2d(dtype=T)`
- `wp.array3d(dtype=T)`
- `wp.array4d(dtype=T)`

对 2D MPM，第一版最常用的是:

```python
x:wp.array(dtype=wp.vec2)          # 粒子位置
v:wp.array(dtype=wp.vec2)          # 粒子速度
C:wp.array(dtype=wp.mat22)         # APIC 仿射速度场
F:wp.array(dtype=wp.mat22)         # deformation gradient
grid_m:wp.array2d(dtype=float)     # 网格质量
grid_v:wp.array2d(dtype=wp.vec2)   # 网格动量或速度
```

Python 侧分配:

```python
x=wp.zeros(n,dtype=wp.vec2,device=device)
grid_m=wp.zeros((nx,ny),dtype=float,device=device)
grid_v=wp.zeros((nx,ny),dtype=wp.vec2,device=device)
```

从 NumPy 初始化:

```python
x_np=np.zeros((n,2),dtype=np.float32)
x=wp.array(x_np,dtype=wp.vec2,device=device)
```

读回 NumPy:

```python
x_host=x.numpy()
```

注意:`array.numpy()` 会同步设备工作。调试时很方便，性能路径里不要每步频繁读回。

## 4. 常用内建函数

标量数学:

- `wp.abs(x)`
- `wp.min(a,b)`
- `wp.max(a,b)`
- `wp.clamp(x,lo,hi)`
- `wp.floor(x)`
- `wp.ceil(x)`
- `wp.sqrt(x)`
- `wp.sin(x)`、`wp.cos(x)`
- `wp.exp(x)`、`wp.log(x)`
- `wp.isnan(x)`、`wp.isinf(x)`、`wp.isfinite(x)`

向量数学:

- `wp.dot(a,b)`
- `wp.cross(a,b)`，主要用于 3D
- `wp.length(v)`
- `wp.length_sq(v)`
- `wp.normalize(v)`
- `wp.outer(a,b)`

矩阵数学:

- `wp.transpose(A)`
- `wp.determinant(A)`
- `wp.inverse(A)`
- `wp.trace(A)`

原子操作:

- `wp.atomic_add(array,index,value)`，一维数组
- `wp.atomic_add(array,i,j,value)`，二维数组
- `wp.atomic_sub(...)`
- `wp.atomic_min(...)`
- `wp.atomic_max(...)`

MPM 的 P2G 会出现很多粒子同时写同一个网格节点，所以必须用 `wp.atomic_add()` 累加质量和动量:

```python
@wp.kernel
def add_mass_kernel(x:wp.array(dtype=wp.vec2),grid_m:wp.array2d(dtype=float),inv_dx:float,mass:float):
    p=wp.tid()
    cell=x[p]*inv_dx
    i=int(wp.floor(cell[0]))
    j=int(wp.floor(cell[1]))
    wp.atomic_add(grid_m,i,j,mass)
```

如果每个线程只写自己的粒子状态，比如 G2P 写 `x[p]`、`v[p]`，不需要 atomic。

## 5. Decorator

### `@wp.kernel`

kernel 是并行入口，只能通过 `wp.launch()` 或 `wp.launch_tiled()` 从 Python 侧启动。

```python
@wp.kernel
def scale_kernel(a:wp.array(dtype=float),s:float):
    i=wp.tid()
    a[i]=a[i]*s
```

规则:

- kernel 参数要标注类型。
- kernel 里可以调用 `wp.tid()`。
- kernel 通常不返回值。
- 结果写进数组参数。
- 第一次 launch 会触发编译，后续会走缓存。

### `@wp.func`

`@wp.func` 是 kernel 内可复用函数。它像 inline helper，不是单独 launch 的 kernel。

```python
@wp.func
def quadratic_weight(x:float):
    ax=wp.abs(x)
    if ax<0.5:
        return 0.75-ax*ax
    elif ax<1.5:
        t=1.5-ax
        return 0.5*t*t
    return 0.0
```

规则:

- 可以被 kernel 调用。
- 不要在 `@wp.func` 里调用 `wp.tid()`；需要线程 id 时，把 id 当参数传进去。
- 可以返回一个值或多个值。
- 可以重载同名函数，只要参数类型不同。

### `@wp.struct`

`@wp.struct` 定义 typed struct，适合集中传仿真参数。

```python
@wp.struct
class SimParams:
    dt:float
    dx:float
    inv_dx:float
    gravity:wp.vec2
    bound:int
```

用法:

```python
@wp.kernel
def grid_update_kernel(grid_v:wp.array2d(dtype=wp.vec2),grid_m:wp.array2d(dtype=float),params:SimParams):
    i,j=wp.tid()
    m=grid_m[i,j]
    if m>0.0:
        v=grid_v[i,j]/m
        v=v+params.gravity*params.dt
        grid_v[i,j]=v
```

### `wp.constant`

`wp.constant()` 用于编译期常量，特别适合固定 loop 大小和 tile shape。

```python
STENCIL=wp.constant(3)

@wp.kernel
def stencil_kernel(out:wp.array(dtype=float)):
    i=wp.tid()
    s=0.0
    for k in range(STENCIL):
        s=s+float(k)
    out[i]=s
```

## 6. 并行模型:不是 Python thread,也不是你手写 OpenMP

Warp 的基本并行模型更像 CUDA kernel。

你不写:

- `threading.Thread`
- `multiprocessing`
- OpenMP pragma

你写:

```python
wp.launch(kernel=my_kernel,dim=n,inputs=[...],device="cuda:0")
```

含义是启动 `n` 个逻辑线程，每个线程执行同一个 kernel body，线程用 `wp.tid()` 区分自己处理哪份数据。

1D launch:

```python
@wp.kernel
def particle_kernel(x:wp.array(dtype=wp.vec2)):
    p=wp.tid()
    x[p]=x[p]+wp.vec2(0.0,-0.01)

wp.launch(kernel=particle_kernel,dim=num_particles,inputs=[x],device=device)
```

2D launch:

```python
@wp.kernel
def clear_grid_kernel(grid_m:wp.array2d(dtype=float),grid_v:wp.array2d(dtype=wp.vec2)):
    i,j=wp.tid()
    grid_m[i,j]=0.0
    grid_v[i,j]=wp.vec2(0.0,0.0)

wp.launch(kernel=clear_grid_kernel,dim=(nx,ny),inputs=[grid_m,grid_v],device=device)
```

3D 和 4D 同理，`wp.tid()` 会返回多个 index。

### MPM 里推荐的 kernel 拆分

第一版显式 2D MPM 可以拆成 4 类 kernel:

```text
clear_grid:   dim=(nx,ny)         每个线程清一个网格节点
p2g:          dim=num_particles   每个线程处理一个粒子,atomic 写 3x3 网格
grid_update:  dim=(nx,ny)         每个线程更新一个网格节点速度和边界
g2p:          dim=num_particles   每个线程从 3x3 网格插值回一个粒子
```

核心判断:

- 多个线程写同一个地址:用 `wp.atomic_add()`。
- 每个线程写自己的地址:直接写。
- 每个线程只读共享数组:直接读。

P2G 是粒子并行，存在写网格冲突，需要 atomic。G2P 是粒子并行，每个粒子只写自己，不需要 atomic。

### CPU/GPU 设备

同一份 kernel 可以 launch 到 CPU 或 CUDA:

```python
wp.launch(kernel=my_kernel,dim=n,inputs=[...],device="cpu")
wp.launch(kernel=my_kernel,dim=n,inputs=[...],device="cuda:0")
```

GPU 上是 CUDA 风格的 SIMT 并行。CPU 上由 Warp runtime 执行已编译代码。入门阶段不要自己在线程层面调度，先把 `dim` 和数据竞争写对。

### Stream 和并发

如果要在同一个 GPU 上重叠计算和拷贝，可以用 stream:

```python
compute_stream=wp.Stream("cuda:0")
transfer_stream=wp.Stream("cuda:0")

with wp.ScopedStream(compute_stream):
    wp.launch(kernel=my_kernel,dim=n,inputs=[a],device="cuda:0")

with wp.ScopedStream(transfer_stream):
    wp.copy(dst,src)
```

普通 MPM 第一版不需要 stream。先写同步、清晰、正确的版本。

## 7. Tile API

Tile 是 Warp 的 block-level cooperative programming API。普通 `wp.launch()` 是“一个逻辑线程处理一个元素”；tile 是“一组线程合作处理一块数据”。

适用场景:

- 行/块 reduction。
- 矩阵乘法。
- FFT/Cholesky 等块级线性代数。
- 需要 shared memory 或 Tensor Core 风格加速的局部块计算。

不适合作为第一版 MPM 的核心工具:

- 3x3 粒子 stencil 很小，用普通 per-thread loop 更直接。
- P2G 的主要问题是跨粒子写冲突，第一版用 atomic 更稳。
- tile 更适合后续优化，比如粒子按网格块排序后做 block-local accumulation。

### `wp.launch_tiled()`

tile kernel 通常用 `wp.launch_tiled()` 启动，并显式给 `block_dim`:

```python
TILE_SIZE=wp.constant(256)
TILE_THREADS=64

@wp.kernel
def row_sum_kernel(a:wp.array2d(dtype=float),out:wp.array2d(dtype=float)):
    row=wp.tid()
    t=wp.tile_load(a[row],TILE_SIZE)
    s=wp.tile_sum(t)
    wp.tile_store(out[row],s)

wp.launch_tiled(
    row_sum_kernel,
    dim=[num_rows],
    inputs=[a,out],
    block_dim=TILE_THREADS,
    device=device,
)
```

这里 `dim=[num_rows]` 不是启动 `num_rows*TILE_THREADS` 个普通逻辑线程，而是启动 `num_rows` 个 tile/block；每个 tile/block 内有 `TILE_THREADS` 个线程合作。

### 2D tile load/store

```python
TILE_M=wp.constant(16)
TILE_N=wp.constant(16)
TILE_THREADS=64

@wp.kernel
def tile_sum_2d_kernel(a:wp.array2d(dtype=float),out:wp.array2d(dtype=float)):
    bi,bj=wp.tid()
    t=wp.tile_load(a,shape=(TILE_M,TILE_N),offset=(bi*TILE_M,bj*TILE_N))
    s=wp.tile_sum(t)
    wp.tile_store(out,s,offset=(bi,bj))

wp.launch_tiled(
    tile_sum_2d_kernel,
    dim=(nx//TILE_M,ny//TILE_N),
    inputs=[a,out],
    block_dim=TILE_THREADS,
    device=device,
)
```

### Tile 常用函数

构造:

- `wp.tile_zeros(shape=...,dtype=...)`
- `wp.tile_empty(shape=...,dtype=...)`
- `wp.tile_ones(shape=...,dtype=...)`
- `wp.tile_full(shape=...,value=...,dtype=...)`

加载/写回:

- `wp.tile_load(...)`
- `wp.tile_store(...)`
- `wp.tile_atomic_add(...)`
- `wp.tile_load_indexed(...)`
- `wp.tile_store_indexed(...)`

map/reduce:

- `wp.tile_sum(t)`
- `wp.tile_min(t)`
- `wp.tile_max(t)`
- `wp.tile_reduce(...)`
- `wp.tile_map(...)`
- `wp.tile_scan_inclusive(...)`
- `wp.tile_scan_exclusive(...)`

线性代数:

- `wp.tile_matmul(a,b,out)`
- `wp.tile_transpose(t)`
- `wp.tile_fft(t)`
- `wp.tile_cholesky(t)`
- `wp.tile_cholesky_solve(...)`

SIMT 和 tile 互转:

- `wp.tile(x)` 把每个线程自己的值组织成一个 tile。
- `wp.untile(t)` 把 tile 元素转回每个线程自己的值。

注意事项:

- tile 操作是 cooperative 的，同一个 block 内线程都要参与。
- 不要让一部分线程跳过 tile 操作，否则可能出同步问题。
- tile shape 通常要用 `wp.constant()`。
- shared memory 有容量限制，tile 不能无限大。
- 部分 tile 线性代数依赖较新的 CUDA Toolkit/MathDx 支持。
- tile stack 这类动态 block-local 数据结构不适合自动微分 backward。

## 8. 自动微分基础

本项目最终要可微，所以要知道 Warp 的 `Tape`。

```python
@wp.kernel
def loss_kernel(x:wp.array(dtype=wp.vec2),target:wp.vec2,loss:wp.array(dtype=float)):
    i=wp.tid()
    d=x[i]-target
    wp.atomic_add(loss,0,wp.dot(d,d))

x=wp.zeros(n,dtype=wp.vec2,device=device,requires_grad=True)
loss=wp.zeros(1,dtype=float,device=device,requires_grad=True)

tape=wp.Tape()

with tape:
    wp.launch(kernel=sim_kernel,dim=n,inputs=[x],device=device)
    wp.launch(kernel=loss_kernel,dim=n,inputs=[x,wp.vec2(0.5,0.5),loss],device=device)

tape.backward(loss)

print(x.grad.numpy())
```

要点:

- 需要求导的数组设置 `requires_grad=True`。
- 上面 `sim_kernel` 是你自己的前向仿真 kernel，占位表示仿真步骤。
- 用 `with wp.Tape():` 记录 forward kernel launch。
- loss 通常是 shape 为 `1` 的标量数组。
- `tape.backward(loss)` 后，梯度在 `array.grad`。
- 如果一个数组在多步仿真里被反复覆盖，梯度可能不正确。多步 MPM 更稳的做法是保存每步状态或用 ping-pong buffer。
- 开发时可以打开 `wp.config.verify_autograd_array_access=True` 检查常见覆盖错误。

## 9. 2D MPM 的推荐入门路线

第一步先写 NumPy baseline，确认算法:

```text
clear_grid()
p2g()
grid_update()
g2p()
```

第二步逐个搬到 Warp:

```text
clear_grid_kernel:  array2d 清零
p2g_kernel:         粒子并行,3x3 stencil,atomic_add 到 grid
grid_update_kernel: 网格并行,质量归一化,重力,边界
g2p_kernel:         粒子并行,从 3x3 grid 插值回粒子
```

第三步再接 Tape:

```text
v0.requires_grad=True
simulate several steps
loss_kernel()
tape.backward(loss)
finite difference check
```

第一版不要急着用 tile。对 MPM 来说，先写清楚普通 kernel、类型、atomic、数组布局，比 tile 更重要。

## 10. 常见坑

- `ModuleNotFoundError:No module named 'warp'`:说明没装 `warp-lang`，或者当前 Python 环境不对。
- 第一次 launch 很慢:正常，Warp 在 JIT 编译 kernel；第二次通常走缓存。
- kernel 里 Python 写法报错:检查是不是用了 list、dict、动态对象、未标注参数或类型不明确表达式。
- 结果全是 0:检查 `wp.launch()` 的 `dim`、`inputs` 顺序、device 是否一致。
- P2G 结果不稳定:检查是否漏了 `wp.atomic_add()`。
- `array.numpy()` 很慢:它会同步和拷贝，调试可以，主循环少用。
- CPU 能跑 GPU 不跑:检查 CUDA 驱动、device 字符串、数组是否分配在同一 device。
- 自动微分梯度不对:检查是否覆盖了还需要反传的数组，先做有限差分。

## 11. 对本项目的直接建议

按这个顺序写代码:

1. `examples/warp_sanity.py`:只做粒子平移，确认安装和 launch。
2. `mpm2d_numpy.py`:NumPy 版 clear/P2G/grid/G2P。
3. `mpm2d_warp.py`:Warp 版四个 kernel。
4. `examples/diff_mpm_smoke.py`:短 horizon，优化初始速度或最终质心。

Warp 版 MPM 的关键不是“怎么开线程”，而是这三件事:

- 用 `dim` 设计并行粒度。
- 用 `wp.tid()` 映射线程到粒子或网格节点。
- 对并发写网格的地方用 atomic。

tile 是后续性能优化工具，不是第一版正确性的前提。

## 参考资料

- Warp Basics:https://nvidia.github.io/warp/stable/user_guide/basics.html
- Warp Built-Ins:https://nvidia.github.io/warp/stable/language_reference/builtins.html
- Warp Differentiability:https://nvidia.github.io/warp/stable/user_guide/differentiability.html
- Warp Tiles:https://nvidia.github.io/warp/stable/user_guide/tiles.html
- Warp Concurrency:https://nvidia.github.io/warp/stable/deep_dive/concurrency.html
