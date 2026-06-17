# Warp 可视化和 USD 输出

结论:本项目当前使用 `MPM-Diff` 环境，Warp 版本是 1.14.0，`wp.render.UsdRenderer` 和 `wp.render.OpenGLRenderer` 都存在。第一版 MPM 推荐先用 `UsdRenderer` 写离线 `.usd` 文件；实时窗口渲染可以后续再接 `OpenGLRenderer`。

本文重点说明六件事:

- 怎么写一个 USD 文件。
- 如果做动画，应该怎么给同一个几何体写 time samples。
- `render_*()` 应该放在 Python scope 还是 Warp kernel scope。
- 常见几何体怎么调用，调用返回什么。
- 怎么用 `pxr.Usd` / `pxr.UsdGeom` 读取 USD。
- 怎么用 Matplotlib 和 imageio 把 USD 数据保存成 PNG/GIF/MP4。

## 1. 当前环境和 API 位置

当前项目环境:

```bash
/home/jerry/miniconda3/envs/MPM-Diff/bin/python
```

检查命令:

```bash
/home/jerry/miniconda3/envs/MPM-Diff/bin/python - <<'PY'
import inspect
import warp as wp
import warp.render as render

print(getattr(wp,"__version__","unknown"))
print(inspect.signature(render.UsdRenderer))
print(hasattr(render,"OpenGLRenderer"))
PY
```

当前 `UsdRenderer` 构造签名:

```python
wp.render.UsdRenderer(stage: str | Usd.Stage,up_axis: str="Y",fps: int=60,scaling: float=1.0)
```

注意新版参数名是 `up_axis`，不是旧版的 `upaxis`。

## 2. 最小 USD 文件怎么写

`UsdRenderer` 会创建或持有一个 USD stage。你传入文件路径，它会创建这个 `.usd` 文件；你调用 `render_*()` 写几何体；最后调用 `save()` 落盘。

最小静态例子:

```python
import os
import numpy as np
import warp as wp
import warp.render as render

os.makedirs("outputs",exist_ok=True)

wp.config.kernel_cache_dir="/tmp/mpmwarp_warp_cache"
wp.init()

renderer=render.UsdRenderer("outputs/static.usd",up_axis="Y",fps=60,scaling=1.0)

points=np.array([
    [-0.5,0.0,0.0],
    [0.0,0.25,0.0],
    [0.5,0.0,0.0],
],dtype=np.float32)

renderer.render_line_strip(
    "line",
    vertices=points,
    color=(0.1,0.4,1.0),
    radius=0.01,
)

renderer.render_sphere(
    "ball",
    pos=(0.0,0.25,0.0),
    rot=(0.0,0.0,0.0,1.0),
    radius=0.05,
    color=(1.0,0.2,0.1),
)

renderer.save()
```

生成的 USD prim 大致在:

```text
/root/line
/root/ball
```

`name` 参数就是 `/root` 下的 prim 名。每次用同一个 `name` 调用，语义上是在更新同一个对象，而不是新建无数个对象。

## 3. 动画 USD 怎么写

动画的核心是 `begin_frame(time_seconds)`。`UsdRenderer` 内部会把秒数乘以 `fps`，得到 USD time code。每一帧对同一个 `name` 调用 `render_*()`，就会写这个 prim 在不同 time code 的属性。

基本结构:

```text
renderer=UsdRenderer(...)
for frame in range(num_frames):
    run_simulation_step()
    renderer.begin_frame(frame/fps)
    renderer.render_*(same_name,...)
    renderer.end_frame()
renderer.save()
```

例子:

```python
import os
import numpy as np
import warp as wp
import warp.render as render

os.makedirs("outputs",exist_ok=True)

wp.config.kernel_cache_dir="/tmp/mpmwarp_warp_cache"
wp.init()

fps=30
renderer=render.UsdRenderer("outputs/line.usd",up_axis="Y",fps=fps,scaling=1.0)

for frame in range(90):
    t=frame/float(fps)
    x=np.linspace(-1.0,1.0,96,dtype=np.float32)
    y=0.25*np.sin(2.0*np.pi*(x-t)).astype(np.float32)
    points=np.stack([x,y,np.zeros_like(x)],axis=1)

    renderer.begin_frame(t)
    renderer.render_line_strip(
        "line",
        vertices=points,
        color=(0.1,0.4,1.0),
        radius=0.01,
    )
    renderer.end_frame()

renderer.save()
```

对 MPM 来说，通常是:

```python
for frame in range(num_frames):
    for substep in range(substeps):
        step()

    points=vec2_to_points3(x)

    renderer.begin_frame(frame/fps)
    renderer.render_points("particles",points,radius=0.003,as_spheres=True)
    renderer.end_frame()
```

如果是三角有限元或三角网格，拓扑不变时第一帧写 indices，后续只更新 points:

```python
renderer.begin_frame(frame/fps)
renderer.render_mesh(
    "tri_fem",
    points=vertices,
    indices=triangles,
    colors=(0.2,0.6,1.0),
    update_topology=(frame==0),
)
renderer.end_frame()
```

如果三角连接关系也变了，必须在变化的帧传 `update_topology=True`。

## 4. render 调用应该写在哪里

`render_*()` 必须写在 Python scope。

正确位置:

```python
for frame in range(num_frames):
    wp.launch(kernel=step_kernel,dim=n,inputs=[...],device=device)
    wp.synchronize()

    points=x.numpy()

    renderer.begin_frame(frame/fps)
    renderer.render_points("particles",points,radius=0.003)
    renderer.end_frame()
```

错误位置:

```python
@wp.kernel
def bad_kernel(...):
    renderer.render_points("particles",points,radius=0.003)
```

原因:

- `@wp.kernel` / `@wp.func` 是 Warp kernel scope，会被 JIT 编译成 CPU/CUDA kernel。
- `wp.render` 是 Python 侧渲染/文件输出 API，依赖 Python 对象、USD stage、pxr 类型。
- USD 写文件是 host 侧行为，不能在 CUDA thread 或 Warp kernel 里调用。

所以实际流程是:

```text
Warp kernel 负责算数据
Python scope 负责读出或传递数据
wp.render 负责写 USD 或画窗口
```

## 5. render 在 CPU 还是 GPU 执行

要区分“仿真数据在哪里”和“render API 在哪里执行”。

`UsdRenderer`:

- `render_*()` 调用在 Python/CPU 端执行。
- USD stage 和文件写入在 CPU 端执行。
- 最稳妥的数据输入是 NumPy array、Python list、tuple、USD/Gf/Vt 类型。
- 如果仿真数组在 GPU 上，比如 `x:wp.array(dtype=wp.vec2,device="cuda:0")`，需要先同步到 CPU，例如 `x.numpy()`。

`OpenGLRenderer`:

- 调用入口仍然在 Python scope。
- 内部 OpenGL buffer 可以和 GPU 交互，部分函数可以接受 `wp.array`。
- 适合实时窗口、headless framebuffer、`get_pixels()` 读 RGB/depth。
- 但它仍然不是 Warp kernel 内函数，不能从 `@wp.kernel` 里调用。

对本项目第一版 MPM:

- 前向仿真可以在 GPU 上。
- 写 USD 建议每隔若干步 `x.numpy()`，再调用 `UsdRenderer`。
- 粒子多时不要每个 substep 都写帧，先每 5 到 20 步写一帧。

2D MPM 转 3D 点:

```python
def vec2_to_xy_points3(x):
    xy=x.numpy()
    points=np.zeros((xy.shape[0],3),dtype=np.float32)
    points[:,0]=xy[:,0]
    points[:,1]=xy[:,1]
    return points
```

放到 XZ 平面:

```python
def vec2_to_xz_points3(x):
    xy=x.numpy()
    points=np.zeros((xy.shape[0],3),dtype=np.float32)
    points[:,0]=xy[:,0]
    points[:,2]=xy[:,1]
    return points
```

## 6. 调用返回什么

当前 `MPM-Diff` 的 `wp.render.UsdRenderer` 返回值如下。

| 调用 | 返回值 | 用途 |
|---|---|---|
| `render.UsdRenderer(path,...)` | `UsdRenderer` 对象 | 持有 USD stage，后续写几何体 |
| `renderer.begin_frame(time)` | `None` | 设置当前 USD time code |
| `renderer.end_frame()` | `None` | USD renderer 中基本是 no-op，保留统一接口 |
| `renderer.save()` | `None` | 保存 stage 到文件 |
| `renderer.render_points(...)` | `Sdf.Path` | 点云/粒子 prim 路径 |
| `renderer.render_mesh(...)` | `Sdf.Path` | mesh prim 路径 |
| `renderer.render_sphere(...)` | `Sdf.Path` | sphere prim 路径 |
| `renderer.render_box(...)` | `Sdf.Path` | cube prim 路径 |
| `renderer.render_plane(...)` | `Sdf.Path` | plane mesh prim 路径 |
| `renderer.render_capsule(...)` | `Sdf.Path` | capsule prim 路径 |
| `renderer.render_cylinder(...)` | `Sdf.Path` | cylinder prim 路径 |
| `renderer.render_cone(...)` | `Sdf.Path` | cone prim 路径 |
| `renderer.render_arrow(...)` | `Sdf.Path` | arrow prim 路径 |
| `renderer.render_line_strip(...)` | `None` | 折线，用 capsule instancer 表示 |
| `renderer.render_line_list(...)` | `None` | 线段列表，用 capsule instancer 表示 |
| `renderer.render_ground(...)` | `None` | 地面 mesh |
| `renderer.render_ref(...)` | `None` | 引用已有 USD asset |

如果你只是在写可视化，大多数时候不需要使用这些返回值。需要后续实例化、检查 USD prim，或用 pxr API 继续改属性时，再保存 `Sdf.Path`。

示例:

```python
mesh_path=renderer.render_mesh("tri_fem",points,indices,update_topology=True)
print(mesh_path)  # /root/tri_fem
```

## 7. 常见几何体怎么写

### 粒子和球

很多粒子:

```python
points=np.random.rand(1000,3).astype(np.float32)

renderer.render_points(
    "particles",
    points=points,
    radius=0.005,
    colors=(0.2,0.5,1.0),
    as_spheres=True,
)
```

`as_spheres=True` 会用 `UsdGeom.PointInstancer` 实例化球，适合看 MPM 粒子。`as_spheres=False` 会写 `UsdGeom.Points`，文件更轻，但 viewer 中显示效果依赖点渲染支持。

单个 ball/sphere:

```python
renderer.render_sphere(
    "ball",
    pos=(0.0,0.5,0.0),
    rot=(0.0,0.0,0.0,1.0),
    radius=0.1,
    color=(1.0,0.2,0.1),
)
```

### 线和弹簧

折线:

```python
vertices=np.array([
    [0.0,0.0,0.0],
    [0.5,0.2,0.0],
    [1.0,0.0,0.0],
],dtype=np.float32)

renderer.render_line_strip(
    "rope",
    vertices=vertices,
    color=(0.1,0.4,1.0),
    radius=0.01,
)
```

线段列表:

```python
vertices=np.array([
    [0.0,0.0,0.0],
    [1.0,0.0,0.0],
    [0.0,1.0,0.0],
    [1.0,1.0,0.0],
],dtype=np.float32)
indices=np.array([0,1,2,3],dtype=np.int32)

renderer.render_line_list(
    "springs",
    vertices=vertices,
    indices=indices,
    color=(0.8,0.3,0.1),
    radius=0.005,
)
```

`render_line_list()` 把 `indices` 每两个一组解释为一条线段: `(0,1)`、`(2,3)`。

### 三角有限元和三角 mesh

三角有限元 surface mesh:

```python
vertices=np.array([
    [0.0,0.0,0.0],
    [1.0,0.0,0.0],
    [1.0,1.0,0.0],
    [0.0,1.0,0.0],
],dtype=np.float32)

triangles=np.array([
    [0,1,2],
    [0,2,3],
],dtype=np.int32)

renderer.render_mesh(
    "tri_fem",
    points=vertices,
    indices=triangles,
    colors=(0.2,0.6,1.0),
    update_topology=True,
)
```

动画时拓扑不变:

```python
for frame in range(num_frames):
    vertices=compute_deformed_vertices(frame)

    renderer.begin_frame(frame/fps)
    renderer.render_mesh(
        "tri_fem",
        points=vertices,
        indices=triangles,
        colors=(0.2,0.6,1.0),
        update_topology=(frame==0),
    )
    renderer.end_frame()
```

`indices` 可以是 flatten 后的 `[0,1,2,0,2,3]`，也可以是 shape `(num_tris,3)` 的数组。Warp 内部会 reshape 成三角形。

### 盒子、平面、地面

盒子:

```python
renderer.render_box(
    "obstacle",
    pos=(0.0,-0.2,0.0),
    rot=(0.0,0.0,0.0,1.0),
    extents=(0.2,0.05,0.2),
    color=(0.4,0.4,0.4),
)
```

平面:

```python
renderer.render_plane(
    "floor_patch",
    pos=(0.0,-0.5,0.0),
    rot=(0.0,0.0,0.0,1.0),
    width=1.0,
    length=1.0,
    color=(0.8,0.8,0.8),
)
```

地面:

```python
renderer.render_ground(size=2.0)
```

`render_ground()` 建议只在初始化时调用一次，不要每帧重复调用。

### 胶囊、圆柱、圆锥、箭头

胶囊:

```python
renderer.render_capsule(
    "capsule",
    pos=(0.0,0.0,0.0),
    rot=(0.0,0.0,0.0,1.0),
    radius=0.05,
    half_height=0.3,
    color=(0.7,0.2,0.9),
)
```

圆柱:

```python
renderer.render_cylinder(
    "cylinder",
    pos=(0.0,0.0,0.0),
    rot=(0.0,0.0,0.0,1.0),
    radius=0.05,
    half_height=0.3,
    color=(0.1,0.8,0.4),
)
```

圆锥:

```python
renderer.render_cone(
    "cone",
    pos=(0.0,0.0,0.0),
    rot=(0.0,0.0,0.0,1.0),
    radius=0.08,
    half_height=0.2,
    color=(0.9,0.6,0.1),
)
```

箭头:

```python
renderer.render_arrow(
    "velocity",
    pos=(0.0,0.0,0.0),
    rot=(0.0,0.0,0.0,1.0),
    base_radius=0.01,
    base_height=0.2,
    cap_radius=0.03,
    cap_height=0.08,
    color=(1.0,0.0,0.0),
)
```

## 8. 什么时候用 OpenGLRenderer

`UsdRenderer` 是离线写文件，适合保存结果、复现实验、用 `usdview`/Omniverse/Blender 打开。

`OpenGLRenderer` 是实时渲染，适合调参时看窗口，或者 headless 读 RGB/depth。

基本实时结构:

```python
import warp as wp
import warp.render as render

wp.init()

renderer=render.OpenGLRenderer(
    title="MPM",
    screen_width=1280,
    screen_height=720,
    fps=60,
    up_axis="Y",
    device="cuda:0",
)

frame=0
while renderer.is_running():
    step()
    points=x.numpy()

    renderer.begin_frame(frame/60.0)
    renderer.render_points("particles",points,radius=0.003,as_spheres=True)
    renderer.end_frame()

    frame=frame+1

renderer.clear()
```

即使用 `OpenGLRenderer`，`render_points()` 这类调用也仍然写在 Python scope，不写进 `@wp.kernel`。

## 9. 如何读取 USD 文件

读取 USD 用 `pxr.Usd` 和 `pxr.UsdGeom`。基本流程:

```python
from pxr import Usd,UsdGeom

stage=Usd.Stage.Open("outputs/line.usd")
if stage is None:
    raise RuntimeError("failed to open USD")

for prim in stage.Traverse():
    print(prim.GetPath(),prim.GetTypeName())
```

这会列出 USD 里所有 prim。常见类型:

- `Mesh`:三角网格、地面、平面。
- `Points`:点云。
- `PointInstancer`:粒子球实例、`render_points(as_spheres=True)`、`render_line_strip()` 的 capsule 实例。
- `BasisCurves`:标准曲线。如果你自己写了曲线 prim，读取最方便。
- `Sphere`、`Cube`、`Capsule`、`Cylinder`、`Cone`:基础几何体。

### 读取动画时间采样

USD 动画不是一张张图片，而是属性在不同 time code 上有 time samples。读取时先拿属性，再拿时间采样:

```python
attr=prim.GetAttribute("points")
times=attr.GetTimeSamples()
print(times)

points_at_last=attr.Get(times[-1])
```

如果属性没有 time samples，读默认值:

```python
value=attr.Get()
```

### 读取标准曲线 `BasisCurves`

如果 USD 里有标准曲线 prim，比如你手动写入的 `/root/line_curve`，读取很直接:

```python
import numpy as np
from pxr import Usd,UsdGeom

stage=Usd.Stage.Open("outputs/line.usd")
prim=stage.GetPrimAtPath("/root/line_curve")
curve=UsdGeom.BasisCurves(prim)

points_attr=curve.GetPointsAttr()
times=points_attr.GetTimeSamples()
time_code=times[-1] if times else Usd.TimeCode.Default()
points=points_attr.Get(time_code)

points_np=np.array([[p[0],p[1],p[2]] for p in points],dtype=np.float32)
```

注意:`wp.render.UsdRenderer.render_line_strip()` 默认不是写 `BasisCurves`，而是写 capsule `PointInstancer`。如果后续要用 Python 方便地反读折线，建议额外写一个 `UsdGeom.BasisCurves`。

### 读取三角网格 `Mesh`

`render_mesh()` 写的是 `UsdGeom.Mesh`。读取顶点和三角形:

```python
import numpy as np
from pxr import Usd,UsdGeom

stage=Usd.Stage.Open("outputs/mesh.usd")
prim=stage.GetPrimAtPath("/root/tri_fem")
mesh=UsdGeom.Mesh(prim)

points_attr=mesh.GetPointsAttr()
times=points_attr.GetTimeSamples()
time_code=times[-1] if times else Usd.TimeCode.Default()

points=points_attr.Get(time_code)
indices=mesh.GetFaceVertexIndicesAttr().Get(time_code)
counts=mesh.GetFaceVertexCountsAttr().Get(time_code)

points_np=np.array([[p[0],p[1],p[2]] for p in points],dtype=np.float32)
indices_np=np.array(indices,dtype=np.int32)
counts_np=np.array(counts,dtype=np.int32)

if np.all(counts_np==3):
    triangles=indices_np.reshape((-1,3))
else:
    raise RuntimeError("not a pure triangle mesh")
```

如果 `update_topology=False`，拓扑属性可能只在默认时间或第一帧有值；读取某一帧拿不到时，退回默认值:

```python
indices=mesh.GetFaceVertexIndicesAttr().Get(time_code)
if indices is None:
    indices=mesh.GetFaceVertexIndicesAttr().Get()
```

### 读取粒子 `PointInstancer`

`render_points(as_spheres=True)` 写的是 `UsdGeom.PointInstancer`。粒子位置在 `positions` 属性:

```python
import numpy as np
from pxr import Usd,UsdGeom

stage=Usd.Stage.Open("outputs/line.usd")
prim=stage.GetPrimAtPath("/root/particles")
instancer=UsdGeom.PointInstancer(prim)

pos_attr=instancer.GetPositionsAttr()
times=pos_attr.GetTimeSamples()
time_code=times[-1] if times else Usd.TimeCode.Default()
positions=pos_attr.Get(time_code)

positions_np=np.array([[p[0],p[1],p[2]] for p in positions],dtype=np.float32)
```

如果 `render_points(as_spheres=False)`，写的是 `UsdGeom.Points`:

```python
points_prim=UsdGeom.Points(prim)
points=points_prim.GetPointsAttr().Get(time_code)
```

### 自动查找第一个可画对象

调试时可以遍历 USD，找到第一个曲线、点云或网格:

```python
from pxr import Usd,UsdGeom

stage=Usd.Stage.Open("outputs/line.usd")

for prim in stage.Traverse():
    if prim.IsA(UsdGeom.BasisCurves):
        print("curve",prim.GetPath())
    elif prim.IsA(UsdGeom.Mesh):
        print("mesh",prim.GetPath())
    elif prim.IsA(UsdGeom.PointInstancer):
        print("instancer",prim.GetPath())
    elif prim.IsA(UsdGeom.Points):
        print("points",prim.GetPath())
```

## 10. 如何用 Matplotlib 可视化 USD

Matplotlib 适合把 USD 中的一帧保存成 PNG。它不是 USD viewer，不会自动理解全部 USD 材质和 instancer 细节；通常做法是用 `pxr` 读出顶点/点坐标，再自己画。

### 读取当前项目粒子并保存 PNG

当前 `test.py` 写的是:

```text
outputs/line.usd
/root/particles
```

它来自 `renderer.render_points("particles",...,as_spheres=True)`，所以读取类型是 `UsdGeom.PointInstancer`:

```python
import os
import numpy as np

os.makedirs("/tmp/mpmwarp_matplotlib_cache",exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR","/tmp/mpmwarp_matplotlib_cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pxr import Usd,UsdGeom

stage=Usd.Stage.Open("outputs/line.usd")
instancer=UsdGeom.PointInstancer(stage.GetPrimAtPath("/root/particles"))

pos_attr=instancer.GetPositionsAttr()
times=pos_attr.GetTimeSamples()
time_code=times[-1] if times else Usd.TimeCode.Default()
positions=pos_attr.Get(time_code)
positions_np=np.array([[p[0],p[1],p[2]] for p in positions],dtype=np.float32)

os.makedirs("outputs",exist_ok=True)
fig=plt.figure(figsize=(7,4),dpi=160)
ax=fig.add_subplot(111)
ax.plot(positions_np[:,0],positions_np[:,1],linewidth=1.5)
ax.scatter(positions_np[:,0],positions_np[:,1],s=10)
ax.set_aspect("equal",adjustable="box")
ax.grid(True,alpha=0.3)
fig.tight_layout()
fig.savefig("outputs/line.png")
plt.close(fig)
```

### 读取标准曲线并保存 PNG

```python
import os
import numpy as np

os.makedirs("/tmp/mpmwarp_matplotlib_cache",exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR","/tmp/mpmwarp_matplotlib_cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pxr import Usd,UsdGeom

stage=Usd.Stage.Open("outputs/line.usd")
curve=UsdGeom.BasisCurves(stage.GetPrimAtPath("/root/line_curve"))

points_attr=curve.GetPointsAttr()
times=points_attr.GetTimeSamples()
time_code=times[-1] if times else Usd.TimeCode.Default()
points=points_attr.Get(time_code)
points_np=np.array([[p[0],p[1],p[2]] for p in points],dtype=np.float32)

os.makedirs("outputs",exist_ok=True)
fig=plt.figure(figsize=(7,4),dpi=160)
ax=fig.add_subplot(111)
ax.plot(points_np[:,0],points_np[:,1],linewidth=2.0)
ax.scatter(points_np[:,0],points_np[:,1],s=8)
ax.set_aspect("equal",adjustable="box")
ax.grid(True,alpha=0.3)
fig.tight_layout()
fig.savefig("outputs/line.png")
plt.close(fig)
```

### 读取普通粒子文件并保存 PNG

```python
import numpy as np
import matplotlib.pyplot as plt
from pxr import Usd,UsdGeom

stage=Usd.Stage.Open("outputs/line.usd")
instancer=UsdGeom.PointInstancer(stage.GetPrimAtPath("/root/particles"))

pos_attr=instancer.GetPositionsAttr()
times=pos_attr.GetTimeSamples()
time_code=times[-1] if times else Usd.TimeCode.Default()
positions=pos_attr.Get(time_code)
positions_np=np.array([[p[0],p[1],p[2]] for p in positions],dtype=np.float32)

plt.figure(figsize=(5,5),dpi=160)
plt.scatter(positions_np[:,0],positions_np[:,1],s=4)
plt.axis("equal")
plt.tight_layout()
plt.savefig("outputs/particles.png")
plt.close()
```

### 读取三角 mesh 并保存 PNG

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from pxr import Usd,UsdGeom

stage=Usd.Stage.Open("outputs/mesh.usd")
mesh=UsdGeom.Mesh(stage.GetPrimAtPath("/root/tri_fem"))

points=mesh.GetPointsAttr().Get()
indices=mesh.GetFaceVertexIndicesAttr().Get()
counts=mesh.GetFaceVertexCountsAttr().Get()

points_np=np.array([[p[0],p[1],p[2]] for p in points],dtype=np.float32)
triangles=np.array(indices,dtype=np.int32).reshape((-1,3))
triangulation=Triangulation(points_np[:,0],points_np[:,1],triangles)

plt.figure(figsize=(5,5),dpi=160)
plt.triplot(triangulation,color="black",linewidth=0.8)
plt.scatter(points_np[:,0],points_np[:,1],s=8)
plt.axis("equal")
plt.tight_layout()
plt.savefig("outputs/mesh.png")
plt.close()
```

## 11. 如何用 imageio 输出 GIF 或 MP4

`imageio` 不直接渲染 USD。它负责把你用 Matplotlib 或 OpenGLRenderer 得到的一帧帧图像写成 GIF/MP4。

典型流程:

```text
pxr 读取 USD 某个 time sample
Matplotlib 画成 RGB image 或 PNG
imageio 把多帧图片写成 GIF/MP4
```

### 当前项目粒子 USD 转 GIF

```python
import os
import numpy as np
import imageio.v2 as imageio

os.makedirs("/tmp/mpmwarp_matplotlib_cache",exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR","/tmp/mpmwarp_matplotlib_cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pxr import Usd,UsdGeom

stage=Usd.Stage.Open("outputs/line.usd")
instancer=UsdGeom.PointInstancer(stage.GetPrimAtPath("/root/particles"))
pos_attr=instancer.GetPositionsAttr()
times=pos_attr.GetTimeSamples()

frames=[]
for time_code in times:
    positions=pos_attr.Get(time_code)
    positions_np=np.array([[p[0],p[1],p[2]] for p in positions],dtype=np.float32)

    fig=plt.figure(figsize=(6,4),dpi=120)
    ax=fig.add_subplot(111)
    ax.plot(positions_np[:,0],positions_np[:,1],linewidth=1.5)
    ax.scatter(positions_np[:,0],positions_np[:,1],s=8)
    ax.set_xlim(float(positions_np[:,0].min())-0.5,float(positions_np[:,0].max())+0.5)
    ax.set_ylim(float(positions_np[:,1].min())-0.5,float(positions_np[:,1].max())+0.5)
    ax.set_aspect("equal",adjustable="box")
    ax.grid(True,alpha=0.3)
    fig.canvas.draw()

    image=np.asarray(fig.canvas.buffer_rgba())[...,:3].copy()
    frames.append(image)
    plt.close(fig)

imageio.mimsave("outputs/line.gif",frames,fps=30)
```

如果你写的是 `BasisCurves`，把读取部分换成:

```python
curve=UsdGeom.BasisCurves(stage.GetPrimAtPath("/root/line_curve"))
points_attr=curve.GetPointsAttr()
times=points_attr.GetTimeSamples()
points=points_attr.Get(time_code)
points_np=np.array([[p[0],p[1],p[2]] for p in points],dtype=np.float32)
```

### USD 曲线转 MP4

```python
imageio.mimsave("outputs/line.mp4",frames,fps=30,quality=8)
```

如果 `imageio` 找不到 ffmpeg，确认环境里有:

```bash
python -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())"
```

### 只把 PNG 序列合成视频

如果你已经有 `outputs/frame_0000.png` 这类图片:

```python
import imageio.v2 as imageio

frames=[]
for i in range(120):
    frames.append(imageio.imread(f"outputs/frame_{i:04d}.png"))

imageio.mimsave("outputs/forward.mp4",frames,fps=30)
```

## 12. 本项目建议

当前 `test.py` 的路线是:

- Warp kernel 生成弹簧链/粒子点。
- Python scope 取出点。
- `UsdRenderer.render_points("particles",...,as_spheres=True)` 写到 `outputs/line.usd` 的 `/root/particles`。

如果你需要“线”的原始折线数据可反读，建议额外写一个标准 `UsdGeom.BasisCurves`；`render_line_strip()` 自己生成的是 capsule instancer，不适合作为数值数据读取入口。

对后续 MPM:

- 粒子:优先 `render_points("particles",points,radius=...)`。
- 三角 FEM/surface:用 `render_mesh("tri_fem",vertices,triangles,update_topology=(frame==0))`。
- 目标点/球形障碍物:用 `render_sphere()`。
- 边界框/障碍物:用 `render_box()`、`render_line_list()`。
- 调试速度/力方向:用 `render_arrow()` 或 `render_line_list()`。

## 13. API 来源

- 当前环境:`/home/jerry/miniconda3/envs/MPM-Diff/lib/python3.12/site-packages/warp/render`
- 当前版本:`warp==1.14.0`
- 相关示例:`test.py` 和 `plot.py`
