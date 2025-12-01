#### 建议使用 Conda 新建独立环境（Python 3.10）：

  - **conda & pip**

    ```bash
    conda create -n sam3-iou python=3.10 -y
    conda activate sam3-iou
    pip install transformers accelerate pillow opencv-python numpy torch torchvision sam3
    ```
  - **确保transformers库包含sam3相关模型**
  ```bash
  python -c "from transformers import Sam3VideoModel"
  ```
  - **如果没有sam3:transformers更新**
  ```bash
  pip install git+https://github.com/huggingface/transformers.git
  or
  pip install --upgrade "git+https://github.com/huggingface/transformers.git"
  ```