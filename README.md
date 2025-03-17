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
pip install -U torch torchvision torch_tensorrt --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.nvidia.com
pip install -U cupy-cuda12x
```


## Installation
```
pip install -U vsrife
```
If you want to download all models at once, run `python -m vsrife`. If you prefer to only download the model you
specified at first run, set `auto_download=True` in `rife()`.

## Usage
```python
from vsdrba import drba_rife

ret = drba_rife(clip)
```

See `__init__.py` for the description of the parameters.


## Benchmarks

| model                | scale | os    | hardware           | arch                                                       | fps 720 | fps 1080 | vram 720 | vram 1080 | backend                                                                  | verified output                    | batch | level | streams | threads | onnx      | onnxslim / onnxsim | onnx shape  | trtexec shape | precision | usage                                                                                               |
|----------------------| ----- | ----- |--------------------|------------------------------------------------------------|---------|----------|----------|-----------| ------------------------------------------------------------------------ | ---------------------------------- | ----- | ----- |---------|---------| --------- | ------------------ | ----------- | ------------- | --------- |-----------------------------------------------------------------------------------------------------|
| rife 4.26 heavy      | 2x    | Linux | 3070laptop / 12400 | [rife](https://github.com/hzwer/Practical-RIFE) (4.26)     | 119     | 53       | 1.6gb    | 3.4gb     | trt 10.8, torch 20241231+cu126, torch_trt 20250102+cu126 (holywu vsrife) | yes, works                         | 1     | 5     | -       | 8       | -         | -                  | -           | static        | RGBH      | rife(clip, trt=True, trt_static_shape=True, model="4.26.heavy", trt_optimization_level=5, sc=False) |
| drba_rife 4.26 heavy | 2x    | Linux | 3070laptop / 12400 | [drba_rife](https://github.com/routineLife1/DRBA) (4.26)   | 158     | 70       | 1.7gb    | 3.7gb     | trt 10.8, torch 20241231+cu126, torch_trt 20250102+cu126 (holywu vsrife) | yes, works                         | 1     | 5     | -       | 8       | -         | -                  | -           | static        | RGBH      | rife(clip, trt=True, trt_static_shape=True, model="4.26.heavy", trt_optimization_level=5, sc=False) |
