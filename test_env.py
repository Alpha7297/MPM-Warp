import os
import sys
import math
import shutil
import subprocess
import importlib
from importlib import metadata

critical_failed=False

def mark(ok,name,msg="",critical=False):
    global critical_failed
    status="PASS" if ok else ("FAIL" if critical else "WARN")
    print(f"[{status}] {name}")
    if msg:
        print(f"       {msg}")
    if critical and not ok:
        critical_failed=True

def version_of(pkg):
    try:
        return metadata.version(pkg)
    except Exception:
        return "unknown"

def check_import(import_name,pkg_name=None,critical=False):
    pkg_name=pkg_name or import_name
    try:
        mod=importlib.import_module(import_name)
        ver=version_of(pkg_name)
        if ver=="unknown":
            ver=getattr(mod,"__version__","unknown")
        mark(True,f"import {import_name}",f"version={ver}",critical)
        return mod
    except Exception as e:
        mark(False,f"import {import_name}",repr(e),critical)
        return None

def check_command(cmd,name,critical=False):
    path=shutil.which(cmd)
    if path is None:
        mark(False,name,f"{cmd} not found in PATH",critical)
        return None
    mark(True,name,path,critical)
    return path

print("========== Basic environment ==========")
mark(sys.version_info[:2]==(3,12),"Python version",sys.version.replace("\n"," "),critical=True)

conda_env=os.environ.get("CONDA_DEFAULT_ENV","")
mark(conda_env=="MPM-Diff","Conda env",f"CONDA_DEFAULT_ENV={conda_env}",critical=False)

print("\n========== Conda/Pip packages ==========")
np=check_import("numpy","numpy",critical=True)
scipy=check_import("scipy","scipy",critical=True)
matplotlib=check_import("matplotlib","matplotlib",critical=True)
check_import("tqdm","tqdm",critical=True)
imageio=check_import("imageio","imageio",critical=True)
imageio_ffmpeg=check_import("imageio_ffmpeg","imageio-ffmpeg",critical=True)
check_import("PIL","Pillow",critical=True)
check_import("pytest","pytest",critical=True)
check_import("ipykernel","ipykernel",critical=True)
check_import("jupyterlab","jupyterlab",critical=True)
check_import("yaml","PyYAML",critical=True)
check_import("pyglet","pyglet",critical=True)
check_import("psutil","psutil",critical=True)
check_import("blosc","blosc",critical=True)
check_import("nvtx","nvtx",critical=False)

print("\n========== System commands ==========")
check_command("ffmpeg","ffmpeg command",critical=False)
check_command("nvidia-smi","nvidia-smi command",critical=False)

try:
    out=subprocess.check_output(
        ["nvidia-smi","--query-gpu=driver_version,name","--format=csv,noheader"],
        text=True,
        timeout=5,
    ).strip()
    mark(True,"NVIDIA driver/GPU",out,critical=False)

    driver=out.split(",")[0].strip()
    major=int(driver.split(".")[0])
    mark(major>=580,"CUDA 13 driver requirement",f"driver={driver},CUDA 13 Warp wheel usually needs driver>=580",critical=False)
except Exception as e:
    mark(False,"NVIDIA driver/GPU",repr(e),critical=False)

print("\n========== Warp ==========")
wp=check_import("warp","warp-lang",critical=True)

if wp is not None:
    try:
        wp.init()
        mark(True,"wp.init()",f"warp version={getattr(wp,'__version__','unknown')}",critical=True)
        try:
            mark(True,"Warp kernel cache",str(wp.config.kernel_cache_dir),critical=False)
        except Exception:
            pass
    except Exception as e:
        mark(False,"wp.init()",repr(e),critical=True)

    @wp.kernel
    def add_kernel(a:wp.array(dtype=wp.float32),b:wp.array(dtype=wp.float32),c:wp.array(dtype=wp.float32)):
        i=wp.tid()
        c[i]=a[i]+b[i]

    def run_warp_kernel(device):
        n=8
        a=wp.array([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0],dtype=wp.float32,device=device)
        b=wp.array([10.0,20.0,30.0,40.0,50.0,60.0,70.0,80.0],dtype=wp.float32,device=device)
        c=wp.zeros(n,dtype=wp.float32,device=device)
        wp.launch(add_kernel,dim=n,inputs=[a,b,c],device=device)
        wp.synchronize()
        out=c.numpy()
        return out

    try:
        out=run_warp_kernel("cpu")
        ok=np is not None and np.allclose(out,[11.0,22.0,33.0,44.0,55.0,66.0,77.0,88.0])
        mark(ok,"Warp CPU kernel",f"out={out}",critical=True)
    except Exception as e:
        mark(False,"Warp CPU kernel",repr(e),critical=True)

    try:
        dev=wp.get_device("cuda:0")
        mark(True,"Warp CUDA device",str(dev),critical=False)
        out=run_warp_kernel("cuda:0")
        ok=np is not None and np.allclose(out,[11.0,22.0,33.0,44.0,55.0,66.0,77.0,88.0])
        mark(ok,"Warp CUDA kernel",f"out={out}",critical=True)
    except Exception as e:
        mark(False,"Warp CUDA kernel",repr(e),critical=True)

    try:
        import warp.render
        mark(True,"import warp.render","ok",critical=True)
    except Exception as e:
        mark(False,"import warp.render",repr(e),critical=True)

print("\n========== USD ==========")
try:
    from pxr import Usd,UsdGeom,Gf
    stage=Usd.Stage.CreateNew("_env_test.usda")
    sphere=UsdGeom.Sphere.Define(stage,"/test_sphere")
    sphere.GetRadiusAttr().Set(1.0)
    sphere.AddTranslateOp().Set(Gf.Vec3d(0.0,0.0,0.0))
    stage.GetRootLayer().Save()
    mark(os.path.exists("_env_test.usda"),"usd-core write test","created _env_test.usda",critical=True)
except Exception as e:
    mark(False,"usd-core write test",repr(e),critical=True)

print("\n========== Image/video utilities ==========")
try:
    exe=imageio_ffmpeg.get_ffmpeg_exe()
    mark(os.path.exists(exe),"imageio-ffmpeg executable",exe,critical=True)
except Exception as e:
    mark(False,"imageio-ffmpeg executable",repr(e),critical=True)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs=[0,1,2,3]
    ys=[0,1,4,9]
    plt.figure()
    plt.plot(xs,ys)
    plt.savefig("_env_test_plot.png")
    plt.close()
    mark(os.path.exists("_env_test_plot.png"),"matplotlib savefig","created _env_test_plot.png",critical=True)
except Exception as e:
    mark(False,"matplotlib savefig",repr(e),critical=True)

try:
    from PIL import Image
    img=Image.new("RGB",(64,64),(255,0,0))
    img.save("_env_test_pillow.png")
    mark(os.path.exists("_env_test_pillow.png"),"Pillow write image","created _env_test_pillow.png",critical=True)
except Exception as e:
    mark(False,"Pillow write image",repr(e),critical=True)

print("\n========== OpenGL display hint ==========")
display=os.environ.get("DISPLAY","")
wayland=os.environ.get("WAYLAND_DISPLAY","")
if display or wayland:
    mark(True,"Display variable",f"DISPLAY={display},WAYLAND_DISPLAY={wayland}",critical=False)
else:
    mark(False,"Display variable","No DISPLAY/WAYLAND_DISPLAY. OpenGLRenderer may fail in headless WSL/SSH,USD output can still work.",critical=False)

print("\n========== Summary ==========")
if critical_failed:
    print("Environment check FAILED.")
    sys.exit(1)
else:
    print("Environment check PASSED.")
    sys.exit(0)
