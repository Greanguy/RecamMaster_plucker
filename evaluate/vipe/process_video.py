# 用法示例:
#   1) 处理单个文件（保持原用法）
#      python process_video.py ./vipe_res_cam01/pose/video0.npz out.json --cam cam01 --prefix frame --inv
#
#   2) 批量处理某个目录下所有 .npz
#      （a）在各自所在目录下生成同名 json：
#          python process_video.py ./vipe_res_cam09/pose --cam cam09 --prefix frame --inv
#
#      （b）把所有 json 输出到指定目录 out_dir（脚本会自动创建）：
#          python process_video.py ./vipe_res_cam01/pose out_dir --cam cam01 --prefix frame --inv
#
import sys, json, argparse
import os
import glob
import numpy as np


def fmt_num(v):
    # 把浮点数格式化成示例风格：
    # - 整数显示为 1 / 33
    # - 小数去掉多余 0：33.900000 -> 33.9
    # - 保留负零信息（-0）
    v = float(v)
    if v == 0.0:
        # 检测负零
        return "-0" if np.signbit(v) else "0"
    s = f"{v:.6f}"
    s = s.rstrip('0').rstrip('.')
    return s


def matrix_to_bracket_string(M):
    # M: (4,4)
    rows = []
    for r in range(4):
        row = M[r, :4]
        rows.append("[" + " ".join(fmt_num(x) for x in row) + "]")
    # 用空格分隔每个 bracket，以匹配示例格式
    return " ".join(rows)


def process_single_npz(npz_path, out_path, cam_name, prefix, do_inv):
    """处理单个 .npz 文件并输出 json。"""
    data = np.load(npz_path, allow_pickle=True)
    keys = list(data.keys())
    if 'data' not in data:
        raise KeyError(f"'data' key not found in {npz_path}. keys: {keys}")
    mats = np.array(data['data'])
    N = mats.shape[0]

    inds = None
    if 'inds' in data:
        inds = np.array(data['inds'])
        # 确保是整型索引（或可转换为 int）
        try:
            inds = [int(x) for x in inds.tolist()]
        except Exception:
            inds = None

    out_dict = {}
    for i in range(N):
        M = mats[i].astype(float)
        if do_inv:
            M = np.linalg.inv(M)
        idx = inds[i] if inds is not None else i
        frame_key = f"{prefix}{idx}"
        out_dict[frame_key] = {
            cam_name: matrix_to_bracket_string(M)
        }

    # 若未指定 out_path，则默认与 npz 同名
    if out_path is None:
        base = os.path.splitext(npz_path)[0]
        out_path = base + f"_{cam_name}.json"

    # 确保输出目录存在
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_dict, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(out_dict)} frames to {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz", help=".npz 文件路径 或 含 .npz 的目录")
    p.add_argument("out", nargs="?", help="输出 json 文件 或 输出目录（当 npz 为目录时）", default=None)
    p.add_argument("--cam",  default="cam11")
    p.add_argument("--prefix", help="frame 前缀 ", default="frame")
    p.add_argument("--inv", help="是否对每个 4x4 矩阵取逆（如果需要转换方向）", action="store_true")
    args = p.parse_args()

    in_path = args.npz

    # 情况 1：输入是目录 -> 批量处理
    if os.path.isdir(in_path):
        npz_files = sorted(glob.glob(os.path.join(in_path, "*.npz")))
        if not npz_files:
            print(f"目录 {in_path} 中没有找到 .npz 文件")
            return

        # 如果给了 out 且不是以 .json 结尾，则当作输出目录
        out_dir = None
        if args.out is not None:
            # 只要不是以 .json 结尾，都认为是目录。脚本会帮你创建。
            if not args.out.lower().endswith(".json"):
                out_dir = args.out
                os.makedirs(out_dir, exist_ok=True)
            else:
                # 如果用户误传了一个 xxx.json，当目录用也行
                out_dir = args.out
                os.makedirs(out_dir, exist_ok=True)

        for npz_path in npz_files:
            base_name = os.path.splitext(os.path.basename(npz_path))[0]
            if out_dir is None:
                # 默认为与 .npz 同目录输出
                out_path = os.path.join(
                    os.path.dirname(npz_path),
                    base_name + f"_{args.cam}.json"
                )
            else:
                out_path = os.path.join(
                    out_dir,
                    base_name + f"_{args.cam}.json"
                )
            process_single_npz(
                npz_path=npz_path,
                out_path=out_path,
                cam_name=args.cam,
                prefix=args.prefix,
                do_inv=args.inv
            )

    # 情况 2：输入是单个文件 -> 保持原有行为
    else:
        out_path = args.out
        process_single_npz(
            npz_path=in_path,
            out_path=out_path,
            cam_name=args.cam,
            prefix=args.prefix,
            do_inv=args.inv
        )


if __name__ == "__main__":
    main()
