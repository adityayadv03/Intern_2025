# 🚀 YOLOv7 Setup Guide (Ubuntu 22.04 + NVIDIA GPU)

This guide will help you set up the YOLOv7 object detection environment on **Ubuntu 22.04** with **GPU acceleration** (NVIDIA RTX series), compatible **CUDA/cuDNN**, and **Python 3.10** using a virtual environment.

---

## 📋 Table of Contents

* [💻 System Requirements](#%ef%b8%8f-system-requirements)
* [📦 Step 1: Check GPU Access](#step-1-check-gpu-access)
* [⚙️ Step 2: Install NVIDIA Driver](#step-2-install-nvidia-driver)
* [🔧 Step 3: Install CUDA and cuDNN](#step-3-install-cuda-and-cudnn)
* [🐍 Step 4: Install Python 3.10 (if needed)](#step-4-install-python-310-if-needed)
* [📁 Step 5: Clone YOLOv7 Repository](#step-5-clone-yolov7-repository)
* [🌐 Step 6: Create Virtual Environment](#step-6-create-virtual-environment)
* [💡 Step 7: Install PyTorch (Compatible with CUDA/cuDNN)](#step-7-install-pytorch-compatible-with-cudacudnn)
* [✅ Step 8: Install YOLOv7 Dependencies](#step-8-install-yolov7-dependencies)
* [📌 Final Checks](#final-checks)
* [📌 References](#references)

---

## 💻 System Requirements

| Component    | Requirement             |
| ------------ | ----------------------- |
| OS           | Ubuntu 22.04 LTS        |
| GPU          | NVIDIA (e.g., RTX 4050) |
| Driver       | NVIDIA Driver >= 525    |
| CUDA Toolkit | CUDA 11.8               |
| cuDNN        | cuDNN 8.6               |
| Python       | Python 3.10             |
| Git, pip     | Installed globally      |

---

## 📦 Step 1: Check GPU Access

Check if your GPU is recognized:

```bash
nvidia-smi
```

If GPU is not detected, continue to install the correct NVIDIA driver.

---

## ⚙️ Step 2: Install NVIDIA Driver

Install recommended version (e.g., **Driver 525**):

```bash
sudo apt update
sudo apt install nvidia-driver-525
```

Reboot your system:

```bash
sudo reboot
```

After reboot, confirm again:

```bash
nvidia-smi
```

---

## 🔧 Step 3: Install CUDA and cuDNN

### 🔹 Why CUDA and cuDNN?

* **CUDA** enables general-purpose GPU computing (needed by PyTorch, TensorFlow, etc.)
* **cuDNN** is a GPU-accelerated library for deep neural networks (used internally by models like YOLOv7)

### ✅ For YOLOv7 and RTX 4050:

* **CUDA version**: 11.8
* **cuDNN version**: 8.6

Install CUDA 11.8 (official method):

```bash
# Download from: https://developer.nvidia.com/cuda-downloads
# Choose Linux > x86_64 > Ubuntu > 22.04 > deb (local)
sudo dpkg -i cuda-repo-ubuntu2204-11-8-local_*.deb
sudo cp /var/cuda-repo-ubuntu2204-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt update
sudo apt install cuda
```

Install cuDNN 8.6:

* Visit: [https://developer.nvidia.com/rdp/cudnn-download](https://developer.nvidia.com/rdp/cudnn-download)
* Select **cuDNN for CUDA 11.x**
* Download `.deb` or `.tgz` and follow install instructions

Update your environment:

```bash
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

---

## 🐍 Step 4: Install Python 3.10 (if needed)

Check current version:

```bash
python3 --version
```

If it's not 3.10, install it:

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev
```

---

## 📁 Step 5: Clone YOLOv7 Repository

Clone official YOLOv7 repo:

```bash
git clone https://github.com/WongKinYiu/yolov7.git
cd yolov7
```

---

## 🌐 Step 6: Create Virtual Environment

Create and activate a Python 3.10 virtual environment:

```bash
python3.10 -m venv yolov7-env
source yolov7-env/bin/activate
```

---

## 💡 Step 7: Install PyTorch (Compatible with CUDA/cuDNN)

Install the compatible version of PyTorch for **CUDA 11.8** and **cuDNN 8.6**:

```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

---

## ✅ Step 8: Install YOLOv7 Dependencies

Install remaining required Python packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📌 Final Checks

Test YOLOv7 runs:

```bash
python detect.py --weights yolov7.pt --source inference/images/horses.jpg
```

If this works, your setup is ready for training or further customization.

---

## 📌 References

* [YOLOv7 GitHub](https://github.com/WongKinYiu/yolov7)
* [CUDA 11.8 Download](https://developer.nvidia.com/cuda-11-8-0-download-archive)
* [cuDNN 8.6](https://developer.nvidia.com/rdp/cudnn-archive)
* [PyTorch CUDA Install](https://pytorch.org/get-started/locally/)

---

> ✅ You are now fully set up with YOLOv7 on a GPU-accelerated Ubuntu 22.04 system!
