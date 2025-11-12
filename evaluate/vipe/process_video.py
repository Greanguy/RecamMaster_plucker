# 用法:
#   python process_video.py ./vipe_res_cam01/pose/video0.npz out.json --cam cam01 --prefix frame --inv
#
import sys, json, argparse
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

def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz", help=".npz path")
    p.add_argument("out", nargs="?", help="输出 json 文件", default=None)
    p.add_argument("--cam",  default="cam11")
    p.add_argument("--prefix", help="frame 前缀 ", default="frame")
    p.add_argument("--inv", help="是否对每个 4x4 矩阵取逆（如果需要转换方向）", action="store_true")
    args = p.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    keys = list(data.keys())
    if 'data' not in data:
        raise KeyError(f"'data' key not found in {args.npz}. keys: {keys}")
    mats = np.array(data['data'])
    N = mats.shape[0]
    inds = None
    if 'inds' in data:
        inds = np.array(data['inds'])
        # 确保是整型索引（或可转换为 int）
        try:
            inds = [int(x) for x in inds.tolist()]
        except:
            inds = None

    out_dict = {}
    for i in range(N):
        M = mats[i].astype(float)
        if args.inv:
            M = np.linalg.inv(M)
        idx = inds[i] if inds is not None else i
        frame_key = f"{args.prefix}{idx}"
        out_dict[frame_key] = {
            args.cam: matrix_to_bracket_string(M)
        }

    out_path = args.out
    if out_path is None:
        base = args.npz.rsplit('.',1)[0]
        out_path = base + f"_{args.cam}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_dict, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(out_dict)} frames to {out_path}")

if __name__ == "__main__":
    main()
