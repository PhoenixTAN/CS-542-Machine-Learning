# Setup GPU for your tensorflow program

## Environment and dependencies
- Windows
- Anaconda
- Jupyter Notebook
- Tensorflow
- CUDA

### Step by step
1. Download & Install the latest version of Anaconda.
2. CUDA Toolkit 9.0 and set up environment variables.
```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\libnvvp
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64
```
3. Restart the machine.
4. Create Tensorflow Environment.
5. Install tensorflow for GPU.
```
conda install tensorflow-gpu
```
6. Validate your installation.

```python
import tensorflow as tf
print(tf.__version__)
1.10.0
```
7. Install Keras.
```
conda install keras
```
8. Validate your installation.
```python
import keras
Using TensorFlow backend.
print(keras.__version__)
2.1.6
```
9. How to check if the code is running on GPU or CPU?

If you are running on the TensorFlow or CNTK backends, your code will automatically run on GPU if any available GPU is detected.

```python
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
```

```
[name: "/device:CPU:0"
device_type: "CPU"
memory_limit: 268435456
locality {
}
incarnation: 8320990378049208634
, name: "/device:GPU:0"
device_type: "GPU"
memory_limit: 4952267161
locality {
  bus_id: 1
  links {
  }
}
incarnation: 2490779436580148339
physical_device_desc: "device: 0, name: GeForce GTX 1060, pci bus id: 0000:01:00.0, compute capability: 6.1"
]
```
### Reference
https://medium.com/@ab9.bhatia/set-up-gpu-accelerated-tensorflow-keras-on-windows-10-with-anaconda-e71bfa9506d1

