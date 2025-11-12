# -*- coding: utf-8 -*-
"""
将示例 JSON 中每个 cam 矩阵按以下流程处理：
1) parse 矩阵字符串为 4x4 矩阵
2) mat = mat.T
# 3) mat = mat[:, [1,2,0,3]]
# 4) mat[:3,1] *= -1
5) mat[:3,3] /= 100
处理成旋转矩阵是单位阵的格式:[1,0,0][0,1,0][0,0,1]
"""

import json
import re
import numpy as np
import argparse
from collections import OrderedDict

# 匹配一个数（包含科学计数法）的正则
_NUM_RE = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?')

def parse_matrix_str(s: str) -> np.ndarray:
    """
    把形如 "[a b c d] [e f g h] [i j k l] [m n o p] " 的字符串解析为 4x4 np.ndarray（float）。
    """
    # 匹配所有方括号内的内容（四行）
    rows = re.findall(r'\[([^\]]+)\]', s)
    if len(rows) < 4:
        raise ValueError("矩阵字符串看起来不是 4 行方括号格式: {!r}".format(s))
    mat = []
    for r in rows[:4]:
        nums = _NUM_RE.findall(r)
        if len(nums) < 4:
            raise ValueError("矩阵行解析不到 4 个数: {!r}".format(r))
        row = [float(x) for x in nums[:4]]
        mat.append(row)
    return np.array(mat, dtype=float)   # shape (4,4)

def matrix_to_string(mat: np.ndarray) -> str:
    """
    将 4x4 矩阵格式化回原 JSON 中的字符串风格：
    "[a b c d] [e f g h] [i j k l] [m n o p]"
    使用简洁数值表示：若为整数则不带小数，否则采用最多 6 位有效数字。
    """
    if mat.shape != (4,4):
        raise ValueError("期望 4x4 矩阵，收到 shape={}".format(mat.shape))

    def fmt(x):
        # 接近整数时显示整数；否则用最多 6 位有效数字
        if abs(x - round(x)) < 1e-9:
            return str(int(round(x)))
        else:
            return ("{:.6g}".format(x)).rstrip('.')
    rows = []
    for r in range(4):
        parts = " ".join(fmt(v) for v in mat[r, :4])
        rows.append(f"[{parts}]")
    return " ".join(rows)

def transform_one_matrix(mat: np.ndarray) -> np.ndarray:
    if mat.shape != (4,4):
        raise ValueError("transform_one_matrix 期望 4x4 输入")
    m = mat.T.copy()
    # m = m[:, [1, 2, 0, 3]].copy()
    # m[:3, 1] *= -1.0
    m[:3, 3] /= 100.0
    return m

def process_json(in_path: str, out_path: str):
    with open(in_path, 'r', encoding='utf-8') as f:
        data = json.load(f, object_pairs_hook=OrderedDict)

    out = OrderedDict()
    # 逐帧逐摄像机处理
    for frame_key, cams in data.items():
        # 保持 frames 的原有顺序/名称
        out_frame = OrderedDict()
        # cams 可能是 dict，key 类似 "cam01","cam02"
        for cam_key, mat_str in cams.items():
            try:
                mat = parse_matrix_str(mat_str)
            except Exception as e:
                # 若解析失败，则把原始字符串直接写回（并继续）
                print(f"WARNING: 解析 {frame_key}/{cam_key} 失败：{e}，将保留原始字符串")
                out_frame[cam_key] = mat_str
                continue

            new_mat = transform_one_matrix(mat)
            new_str = matrix_to_string(new_mat)
            out_frame[cam_key] = new_str

        out[frame_key] = out_frame

    # 写出 JSON（保证可读）
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="处理相机外参 JSON 并写出新的 JSON。")
    parser.add_argument('--input', '-i', required=True, help="输入 JSON 路径（原始）")
    parser.add_argument('--output', '-o', required=True, help="输出 JSON 路径（处理后）")
    args = parser.parse_args()
    process_json(args.input, args.output)
    print(f"处理完成，已写入：{args.output}")
