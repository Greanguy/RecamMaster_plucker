import torch
from torchvision.ops import box_iou

# boxes shape: [N, 4], [M, 4]，格式 (x1, y1, x2, y2)
boxes1 = torch.tensor([[50, 50, 150, 150]], dtype=torch.float32)
boxes2 = torch.tensor([[60, 60, 140, 140],
                       [0,   0, 100, 100]], dtype=torch.float32)

ious = box_iou(boxes1, boxes2)   # [1, 2]
print(ious)
