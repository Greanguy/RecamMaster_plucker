#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute RotErr / TransErr from two camera trajectory JSONs.

只需保证输入的轨迹参数是标准形式即可

Input JSON format example:
{
  "frame0": { "cam01": "[r00 r01 r02 tx] [r10 r11 r12 ty] [r20 r21 r22 tz] [0 0 0 1]" },
  "frame1": { "cam01": "..." },
  ...
}

Usage:

# 单个 EST
python cam_err.py \
  --gt path/to/gt_traj.json \
  --est path/to/est_traj.json \
  --key cam01 \
  --convention cam2world   # or world2cam

# 批量模式：对 est_dir 中所有 .json 估计文件逐个评估，写出 CSV
python cam_err.py \
  --gt path/to/gt_traj.json \
  --est_dir path/to/est_dir \
  --key cam01 \
  --convention cam2world \
  --csv_out results/cam_err_all.csv \
  --ignore_first
"""

import json, re, argparse, math, csv
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt', required=True, help='GT 轨迹 JSON 路径')

    # 单个 or 批量：二选一
    ap.add_argument('--est', default=None,
                    help='单个 EST 轨迹 JSON 路径（与 --est_dir 二选一）')
    ap.add_argument('--est_dir', default=None,
                    help='若提供，则对该目录下所有 .json 作为 EST 批量评估')

    ap.add_argument('--key', default='cam09', help='相机键名 如 cam09')
    ap.add_argument('--convention', default='world2cam',
                    choices=['world2cam','cam2world'],
                    help='外参约定: world2cam 表示 x_cam=R x_world + t; cam2world 表示 x_world=R x_cam + t')
    ap.add_argument('--ignore_first', action='store_true',
                    help='是否忽略首帧统计 常见做法，默认不忽略')

    # 单文件模式下：输出 JSON
    ap.add_argument('--out', default=None,
                    help='可选：单文件模式下把结果保存为 JSON 文件（会自动创建目录）')

    # 批量模式下：输出 CSV
    ap.add_argument('--csv_out', default=None,
                    help='批量模式下，把所有 EST 的结果写入 CSV 文件路径')

    return ap.parse_args()

_num_re = re.compile(r'[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?')

def _string_4x4_to_mat(s: str) -> np.ndarray:
    """把形如 "[..] [..] [..] [..]" 的一行转为 4x4 float 矩阵（按行展开）。"""
    vals = [float(x.group(0)) for x in _num_re.finditer(s)]
    if len(vals) != 16:
        raise ValueError(f'Expect 16 numbers, got {len(vals)} in: {s}')
    M = np.array(vals, dtype=np.float64).reshape(4,4)
    return M

def _project_to_so3(R: np.ndarray) -> np.ndarray:
    """用 SVD 投影到最近的正交矩阵(保证 det=+1)"""
    U, S, Vt = np.linalg.svd(R)
    R_ = U @ Vt
    if np.linalg.det(R_) < 0:
        U[:, -1] *= -1
        R_ = U @ Vt
    return R_

def load_traj(json_path: str, key: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """返回列表 R_list, t_list; 按 frame0..frameN 排序"""
    with open(json_path, 'r') as f:
        J: Dict[str, Dict[str, str]] = json.load(f)
    # 按帧序排序
    items = sorted(J.items(), key=lambda kv: int(kv[0].replace('frame','')))
    R_list, t_list = [], []
    for _, cam_dict in items:
        if key not in cam_dict:
            raise KeyError(f'Key "{key}" not found in a frame.')
        M = _string_4x4_to_mat(cam_dict[key])
        R = _project_to_so3(M[:3,:3])
        t = M[:3, 3].astype(np.float64)
        R_list.append(R)
        t_list.append(t)
    return R_list, t_list

def camera_center(R: np.ndarray, t: np.ndarray, convention: str) -> np.ndarray:
    """根据约定返回相机中心 C(世界坐标系)"""
    if convention == 'world2cam':
        # x_cam = R x_world + t  =>  C = -R^T t
        return -R.T @ t
    else:
        # x_world = R x_cam + t  =>  C = t
        return t

def relative_trajectory(Rs: List[np.ndarray], Cs: List[np.ndarray]):
    """把轨迹相对化到首帧(R_rel[i]=R_i R_0^T; C_rel[i]=C_i - C_0)"""
    R0 = Rs[0]
    C0 = Cs[0]
    R_rel = [R @ R0.T for R in Rs]
    C_rel = [C - C0 for C in Cs]
    return R_rel, C_rel

def rotation_error_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    """角轴距离（度）。"""
    R = R_est @ R_gt.T
    tr = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = math.degrees(math.acos(tr))
    return float(theta)

def align_scale_least_squares(C_est_rel: List[np.ndarray], C_gt_rel: List[np.ndarray]) -> float:
    """最小二乘求 s: min_s sum || C_gt - s C_est ||^2  => s = (Σ est·gt) / (Σ est·est)"""
    num = 0.0
    den = 0.0
    for ce, cg in zip(C_est_rel, C_gt_rel):
        num += float(ce @ cg)
        den += float(ce @ ce)
    if den <= 1e-12:
        return 1.0
    return num / den

def evaluate(gt_json: str, est_json: str, key: str, convention: str, ignore_first: bool):
    # 读取
    R_gt, t_gt = load_traj(gt_json, key)
    R_est, t_est = load_traj(est_json, key)
    assert len(R_gt) == len(R_est), "GT 与 EST 帧数不一致"
    N = len(R_gt)

    # 相机中心
    C_gt = [camera_center(R_gt[i], t_gt[i], convention) for i in range(N)]
    C_est = [camera_center(R_est[i], t_est[i], convention) for i in range(N)]

    # 轨迹相对化
    Rgt_rel, Cgt_rel = relative_trajectory(R_gt, C_gt)
    Rest_rel, Cest_rel = relative_trajectory(R_est, C_est)

    # 尺度对齐（仅影响平移误差）
    s = align_scale_least_squares(Cest_rel, Cgt_rel)
    Cest_rel = [s * c for c in Cest_rel]

    # 计算逐帧误差
    idx_start = 1 if ignore_first else 0
    rot_err_list = []
    trans_err_list = []
    for i in range(idx_start, N):
        rot_err_list.append(rotation_error_deg(Rest_rel[i], Rgt_rel[i]))
        trans_err_list.append(float(np.linalg.norm(Cest_rel[i] - Cgt_rel[i])))

    def stats(x):
        x = np.array(x, dtype=np.float64)
        return dict(mean=float(x.mean()), median=float(np.median(x)),
                    min=float(x.min()), max=float(x.max()))

    return {
        'num_frames': N,
        'ignore_first': ignore_first,
        'scale_s': float(s),
        'RotErr_deg': {**stats(rot_err_list)},
        'TransErr':   {**stats(trans_err_list)},
        # per_frame 在批量 CSV 中不会用到，但单个分析时可能还想看
        'per_frame': {
            'rot_err_deg': rot_err_list,
            'trans_err': trans_err_list
        }
    }

def run_single(args):
    """单个 EST 文件模式（旧行为）"""
    if args.est is None:
        raise SystemExit('单文件模式下必须提供 --est')

    out = evaluate(args.gt, args.est, args.key, args.convention, args.ignore_first)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open('w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f'[cam_err] 结果已保存到: {str(out_path.resolve())}')
    else:
        import pprint
        pprint.pprint(out)

def run_batch(args):
    """批量模式：对 est_dir 中所有 .json 文件评估并写出 CSV"""
    if args.est_dir is None:
        raise SystemExit('批量模式下必须提供 --est_dir')

    est_dir = Path(args.est_dir)
    if not est_dir.is_dir():
        raise SystemExit(f'est_dir 不是目录: {est_dir}')

    est_paths = sorted(est_dir.glob('*.json'))
    if not est_paths:
        raise SystemExit(f'est_dir 中未找到任何 .json 文件: {est_dir}')

    if args.csv_out is None:
        raise SystemExit('批量模式下必须通过 --csv_out 指定输出 CSV 文件路径')

    csv_path = Path(args.csv_out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    all_rot_mean = []
    all_trans_mean = []

    for est_path in est_paths:
        res = evaluate(args.gt, str(est_path), args.key, args.convention, args.ignore_first)

        name = est_path.name  # 可视为“每个视频”的名字
        rot_stats = res['RotErr_deg']
        trans_stats = res['TransErr']

        rows.append({
            'name': name,
            'num_frames': res['num_frames'],
            'ignore_first': res['ignore_first'],
            'scale_s': res['scale_s'],
            'RotErr_mean': rot_stats['mean'],
            'RotErr_median': rot_stats['median'],
            'RotErr_min': rot_stats['min'],
            'RotErr_max': rot_stats['max'],
            'TransErr_mean': trans_stats['mean'],
            'TransErr_median': trans_stats['median'],
            'TransErr_min': trans_stats['min'],
            'TransErr_max': trans_stats['max'],
        })

        all_rot_mean.append(rot_stats['mean'])
        all_trans_mean.append(trans_stats['mean'])

    # 所有“视频”的平均误差：这里用“各视频 mean 的平均”
    all_rot_mean = float(np.mean(all_rot_mean))
    all_trans_mean = float(np.mean(all_trans_mean))

    overall_row = {
        'name': 'OVERALL',
        'num_frames': '',
        'ignore_first': args.ignore_first,
        'scale_s': '',
        'RotErr_mean': all_rot_mean,
        'RotErr_median': '',
        'RotErr_min': '',
        'RotErr_max': '',
        'TransErr_mean': all_trans_mean,
        'TransErr_median': '',
        'TransErr_min': '',
        'TransErr_max': '',
    }
    rows.append(overall_row)

    fieldnames = [
        'name', 'num_frames', 'ignore_first', 'scale_s',
        'RotErr_mean', 'RotErr_median', 'RotErr_min', 'RotErr_max',
        'TransErr_mean', 'TransErr_median', 'TransErr_min', 'TransErr_max',
    ]

    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f'[cam_err] 批量结果已保存到 CSV: {csv_path.resolve()}')
    print(f'[cam_err] OVERALL RotErr_mean={all_rot_mean:.4f} deg, '
          f'TransErr_mean={all_trans_mean:.4f}')

if __name__ == '__main__':
    args = parse_args()

    # 选择模式
    has_est = args.est is not None
    has_est_dir = args.est_dir is not None

    if not has_est and not has_est_dir:
        raise SystemExit('必须至少提供 --est 或 --est_dir 之一')

    # 如果两个都提供，优先认为是批量模式（--est 可以忽略）
    if has_est_dir:
        run_batch(args)
    else:
        run_single(args)
