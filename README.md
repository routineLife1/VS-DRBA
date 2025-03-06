# VS-DRBA
Distance Ratio Based Adjuster for Animeinterp, based on https://github.com/routineLife1/DRBA and https://github.com/HolyWu/vs-rife.

> This project is modified from [HolyWu/vs-rife](https://github.com/HolyWu/vs-rife) and achieves nearly the same interpolation quality as the original [DRBA](https://github.com/routineLife1/DRBA) project.
> 
> With TensorRT integration, it achieves a 400% acceleration, enabling real-time playback on high-performance NVIDIA GPUs.
>

## Dependencies
- [PyTorch](https://pytorch.org/get-started/) 2.6.0 or later
- [VapourSynth](http://www.vapoursynth.com/) R66 or later
- [vs-miscfilters-obsolete](https://github.com/vapoursynth/vs-miscfilters-obsolete) (only needed for scene change detection)
- [cupy](https://github.com/cupy/cupy) cupy-cuda11x or later (critical for acceleration)

`trt` requires additional packages:
- [TensorRT](https://developer.nvidia.com/tensorrt) 10.7.0.post1 or later
- [Torch-TensorRT](https://pytorch.org/TensorRT/) 2.6.0 or later

To install the latest stable version of PyTorch, Torch-TensorRT and cupy, run:
```
pip install -U packaging setuptools wheel
pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install --no-deps -U torch_tensorrt --index-url https://download.pytorch.org/whl/cu126
pip install -U tensorrt --extra-index-url https://pypi.nvidia.com
pip install -U cupy-cuda12x
```


## Installation
```
pip install -U numpy
pip install -U tqdm
pip install -U requests
pip install VapourSynth
```


## Usage
```python
from vsrife import rife

ret = rife(clip)
```

See `__init__.py` for the description of the parameters.
