# Benchmark TensorRT with FP32-FP16-INT8
Benchmark inference speed of CNNs with various quantization methods with TensorRT

# Install TensorRT
+ Download TensorRT from [NVIDIA](https://developer.nvidia.com/tensorrt-getting-started).
+ Create a conda environment.
+ Then
```
sudo dpkg -i nv-tensorrt-repo-ubuntu1604-cuda11.1-trt7.2.3.4-ga-20210226_1-1_amd64.deb
sudo apt-key add /var/nv-tensorrt-repo-ubuntu1604-cuda11.1-trt7.2.3.4-ga-20210226/7fa2af80.pub
sudo apt-get update
sudo apt-get install tensorrt
sudo apt-get install python3-libnvinfer-dev
~/anaconda3/envs/pytorch/bin/pip install nvidia-pyindex
~/anaconda3/envs/pytorch/bin/pip install nvidia-tensorrt
~/anaconda3/envs/pytorch/bin/pip  install termcolor
```