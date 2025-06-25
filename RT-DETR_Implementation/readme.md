
# RT-DETR Implementation 🚀

This repository contains the implementation of **RT-DETR (Real-Time Detection Transformer)** using the Ultralytics framework. RT-DETR is an anchor-free, transformer-based object detection model capable of high accuracy with real-time performance.

---


## 🧠 Learning Resources

This implementation was guided by the following official resources:
- 📘 [Ultralytics RT-DETR Documentation](https://docs.ultralytics.com/models/rtdetr/)
- 🎥 [RT-DETR YouTube Tutorial](https://youtu.be/SArFQs6CHwk)

---

## ⚙️ System Configuration

- **GPU:** NVIDIA RTX 4050  
- **CUDA Version:** 11.8  
- **Python Version:** 3.10  
- **OS:** Ubuntu 22.04  
- **Framework:** Ultralytics + PyTorch

---

## 📂 File Structure

```
RT-DETR_Implementation/
├── rtdetr_train_infer.py         # Basic inference script
├── rtdetr_benchmark_final.py     # Annotated inference with FPS, CPU, GPU, RAM
├── requirements.txt              # All required dependencies
├── rtdetr-l.pt                   # Pretrained RT-DETR-L model (optional)
└── inference/Video/New_york.mp4  # Test video input
```

---

## 🛠️ Setup Instructions

1. **Clone the repository:**
```bash
git clone https://github.com/adityayadv03/Intern_2025.git
cd Intern_2025/RT-DETR_Implementation
```

2. **Create and activate a virtual environment:**
```bash
python3.10 -m venv rtdetr-env
source rtdetr-env/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Run the inference script:**
```bash
python inference_script.py
```

---
