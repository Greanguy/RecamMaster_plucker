# Evaluate 

本模块用于对 **ReCamMaster** 生成结果进行**自动化性能评估**，覆盖视觉质量、相机精度及多视角同步性三大类指标。

## 评估指标体系

### 一、视觉质量指标（Visual Quality）

该部分用于综合评估生成视频在**空间与时间维度**的质量与一致性，主要包括：

#### 1、 CLIP-T / CLIP-F：
- **CLIP-T（Text Consistency）**：逐帧提取图像嵌入与文本提示嵌入，计算余弦相似度并取平均，反映生成视频的语义一致性。  
- **CLIP-F（Frame Consistency）**：对相邻帧嵌入计算余弦相似度并平均，反映时间平滑性与局部时序稳定。

#### 2、FID / FVD：
- **FID (Fréchet Inception Distance)**：衡量生成视频帧分布与真实视频帧分布在 Inception 特征空间中的差距。  
- **FVD (Fréchet Video Distance)**：基于 I3D 特征的时序版本，反映整体视频流的时间一致性与真实感。

#### 3、VBench：
VBench 提供多维度的视频生成质量评估，本模块其评估视觉质量中的六个维度：
- `subject_consistency`：主体外观稳定性  
- `background_consistency`：背景一致性  
- `motion_smoothness`：运动平滑度  
- `temporal_flickering`：时间闪烁稳定性  
- `aesthetic_quality`：美学质量  
- `imaging_quality`：成像细节与清晰度  

### 二、相机精度指标（Camera Accuracy）

- **RotErr（Rotation Error）**：预测相机姿态与真实外参间的旋转误差（单位 °）；
- **TransErr（Translation Error）**：平移向量差的 L2 范数，反映轨迹精度。

该模块用于验证 ReCamMaster 在重建相机位姿时的几何精度。


### 三、多视角同步性指标（View Synchronization）

用于评估生成模型在不同视角下保持一致性的能力，包括：
- **CLIP-V**：对同一时间戳 t ，取不同视角帧的CLIP 图像嵌入做余弦相似度并按视角对/时间求平均；
- **FVD-V**：在固定时间戳上沿视角轴把多视图帧视作“视频序列”，用 I3D 特征计算 FVD，反映跨视角的一致性；
- **Mat.Pix. (GIM)**：同一时间戳的源视角帧 vs 目标视角帧，运行 GIM 进行像素匹配，统计置信度≥阈值的匹配像素比例，衡量跨视角几何一致性。

此类指标反映模型在多相机同步生成中的跨视角时序与语义对齐效果。

---

## Usage

### CLIP-T / CLIP-F / CLIP-V：  
  参见 `clip-t-f.sh`，其中包含该命令文件的运行以及参数说明 

### VBench：  
  - **Install vbench with pip**

    ```bash
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 # or any other PyTorch version with CUDA<=12.1
    pip install vbench
    ```

    然后即可按照vbench.sh中的命令以及参数说明来计算评估指标了

### FID / FVD：  
  - **Installation**
    ```bash
    cd RecamMaster_plucker/evaluate/fid-metrics
    pip install -r requirements.txt
    ```

    然后按照fid-fvd.sh中的说明即可

### RotErr / TranErr：  
  - **Install vipe**

  按照/RecamMaster_plucker/evaluate/vipe中的说明安装vipe/配置所需环境，然后通过运行cam_pose.sh来得到生成视频的外参估计
  随后注意将ground true的外参矩阵处理成vipe所提取的相同格式(c2w矩阵,旋转矩阵由单位矩阵起始)，例如按./cam_err/process2.sh中处理的那样
  最后按照cam_err.sh的说明运行即可

### Mat.Pix.： 
  - **Install GIM**

  建议先按照RecamMaster_plucker/evaluate/gim/environment.yaml为gim配置一个虚拟环境，然后在该环境下运行mat.pix.sh，该命令文件中包含参数的说明

---
## Test

