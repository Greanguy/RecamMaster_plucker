import os
import argparse
from typing import List, Tuple, Optional
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

# ---------------- Video backends ----------------
_BACKENDS = {}

def _init_decord():
    try:
        import decord
        decord.bridge.set_bridge("torch")
        _BACKENDS["decord"] = decord
        return True
    except Exception:
        return False

def _init_cv2():
    try:
        import cv2
        _BACKENDS["cv2"] = cv2
        return True
    except Exception:
        return False

_HAS_DEC = _init_decord()
_HAS_CV2 = _init_cv2() if not _HAS_DEC else True
if not (_HAS_DEC or _HAS_CV2):
    raise RuntimeError("Need at least one video backend: install `decord` or `opencv-python`.")

# ---------------- OpenCLIP ----------------
import open_clip

def load_clip_online(model_name="ViT-L-14", pretrained="openai", device="cuda", fp16=False):
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    if fp16 and device.startswith("cuda"):
        model = model.to(dtype=torch.float16)
    return model, tokenizer, preprocess

def load_clip_offline(model_name, pretrained_path, device="cuda", fp16=False):
    """
    Load CLIP model without any network access.
    `pretrained_path` can be a .safetensors or .pt/.bin state dict file.
    """
    # build empty model + transforms
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=None, device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()

    # load state dict
    pretrained_path = os.path.expanduser(pretrained_path)
    if not os.path.isfile(pretrained_path):
        raise FileNotFoundError(f"Pretrained weights not found: {pretrained_path}")

    state = None
    if pretrained_path.endswith(".safetensors"):
        from safetensors.torch import load_file as safe_load_file
        state = safe_load_file(pretrained_path)
    else:
        state = torch.load(pretrained_path, map_location="cpu")

    # some checkpoints wrap in {'state_dict': ...}
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]

    # strip possible 'model.' prefix
    new_state = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("model."):
            nk = nk[len("model."):]
        new_state[nk] = v

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        print(f"[load_clip_offline] Warning: missing keys: {len(missing)} (showing first 5) -> {missing[:5]}")
    if unexpected:
        print(f"[load_clip_offline] Warning: unexpected keys: {len(unexpected)} (showing first 5) -> {unexpected[:5]}")

    if fp16 and device.startswith("cuda"):
        model = model.to(dtype=torch.float16)

    return model, tokenizer, preprocess

# ---------------- Sampling ----------------
def sample_indices(n_total: int, every_n: Optional[int], num_frames: Optional[int]) -> np.ndarray:
    assert not (every_n and num_frames), "Use either --every-n or --num-frames, not both."
    if n_total <= 0:
        return np.array([], dtype=np.int64)
    if every_n:
        idx = np.arange(0, n_total, every_n, dtype=np.int64)
        return idx if len(idx) else np.array([0], dtype=np.int64)
    if num_frames:
        if num_frames >= n_total:
            return np.arange(0, n_total, dtype=np.int64)
        xs = np.linspace(0, n_total - 1, num_frames)
        return np.floor(xs + 1e-6).astype(np.int64)
    return np.arange(0, n_total, dtype=np.int64)

# ---------------- Video decoding ----------------
def read_video_frames_decord(path: str, indices: np.ndarray) -> List["torch.Tensor"]:
    dec = _BACKENDS["decord"]
    vr = dec.VideoReader(path)
    n_total = len(vr)
    indices = indices[indices < n_total]
    if len(indices) == 0:
        return []
    frames = vr.get_batch(indices)  # torch uint8 [T,H,W,3]
    frames = frames.permute(0, 3, 1, 2).contiguous().float() / 255.0  # [T,3,H,W]
    return [frames[i] for i in range(frames.shape[0])]

def read_video_frames_cv2(path: str, indices: np.ndarray) -> List["torch.Tensor"]:
    cv2 = _BACKENDS["cv2"]
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        frame = frame[:, :, ::-1]  # BGR->RGB
        frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        frames.append(frame)
    cap.release()
    return frames

def read_video_frames(path: str, every_n: Optional[int], num_frames: Optional[int]) -> List["torch.Tensor"]:
    if "decord" in _BACKENDS:
        dec = _BACKENDS["decord"]
        vr = dec.VideoReader(path)
        n_total = len(vr)
        idx = sample_indices(n_total, every_n, num_frames)
        return read_video_frames_decord(path, idx)
    else:
        cv2 = _BACKENDS["cv2"]
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return []
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        idx = sample_indices(n_total, every_n, num_frames)
        return read_video_frames_cv2(path, idx)

# ---------------- CLIP-T / CLIP-F ----------------
@torch.no_grad()
def compute_text_feat(model, tokenizer, text: str, device="cuda", fp16=False) -> torch.Tensor:
    tokens = tokenizer([text]).to(device)
    feat = model.encode_text(tokens)
    if fp16 and device.startswith("cuda"):
        feat = feat.to(dtype=torch.float16)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.squeeze(0)

from torchvision import transforms as T

@torch.no_grad()
def compute_image_feats(model, preprocess, frames: List[torch.Tensor],
                        device="cuda", fp16=False, batch_size=64) -> torch.Tensor:
    if len(frames) == 0:
        return torch.empty(0, 1, device=device)

    to_pil = T.ToPILImage()  # 将 Tensor(C,H,W) 转 PIL.Image，支持 float[0,1]
    # 先把 Tensor -> PIL，再走 open_clip 的 preprocess（resize/center-crop/normalize等）
    imgs = [preprocess(to_pil(t.clamp(0, 1))) for t in frames]

    X = torch.stack(imgs, dim=0).to(device)
    feats = []
    n = X.shape[0]
    for i in range(0, n, batch_size):
        xb = X[i:i+batch_size]
        fb = model.encode_image(xb)
        if fp16 and device.startswith("cuda"):
            fb = fb.to(dtype=torch.float16)
        fb = fb / fb.norm(dim=-1, keepdim=True)
        feats.append(fb)
    return torch.cat(feats, dim=0)

def clip_T_and_F(model, tokenizer, preprocess, frames: List[torch.Tensor], prompt: str,device="cuda", fp16=False, batch_size=64):
    if len(frames) == 0:
        return float("nan"), float("nan"), np.array([]), np.array([])
    text_feat = compute_text_feat(model, tokenizer, prompt, device=device, fp16=fp16)  # [D]
    img_feats = compute_image_feats(model, preprocess, frames, device=device, fp16=fp16, batch_size=batch_size)  # [T,D]
    sim_t = (img_feats @ text_feat)                                # [T]
    clip_t = sim_t.mean().item()
    if img_feats.shape[0] >= 2:
        sim_pairs = (img_feats[:-1] * img_feats[1:]).sum(dim=-1)   # [T-1]
        clip_f = sim_pairs.mean().item()
        per_pair = sim_pairs.float().cpu().numpy()
    else:
        clip_f, per_pair = float("nan"), np.array([])
    return clip_t, clip_f, sim_t.float().cpu().numpy(), per_pair

# ---------------- CSV loader ----------------
def load_pairs_from_csv(csv_path: str, video_col: str, text_col: str,video_root: Optional[str]) -> List[Tuple[str, str]]:
    df = pd.read_csv(csv_path)
    assert video_col in df.columns and text_col in df.columns, \
        f"CSV must have columns: {video_col},{text_col}. Got: {list(df.columns)}"
        
     # 获取 video_root 文件夹中所有视频文件（按文件名排序）
    assert video_root is not None and os.path.isdir(video_root), f"Invalid video_root: {video_root}"
    video_files = sorted([
        os.path.join(video_root, f)
        for f in os.listdir(video_root)
        if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    ])

    # 取前 len(df) 个视频文件路径与 CSV 的文本列一一对应
    if len(video_files) < len(df):
        raise ValueError(f"视频数量不足：视频文件 {len(video_files)} 个，但 CSV 有 {len(df)} 条记录。")
    pairs = []
    for i, (_, row) in enumerate(df.iterrows()):
        prompt = str(row[text_col])
        video_path = video_files[i]
        pairs.append((video_path, prompt))
    return pairs

# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate CLIP-T and CLIP-F for videos (offline-friendly).")

    g = parser.add_argument_group("Inputs")
    g.add_argument("--video", type=str, default=None, help="Single video path (ignored if --prompts-csv given)")
    g.add_argument("--prompt", type=str, default=None, help="Text prompt for the single video")
    g.add_argument("--prompts-csv", type=str, default=None, help="CSV with columns: file_name,text (default)")
    g.add_argument("--csv-video-col", type=str, default="file_name",help="CSV column name for video name (default 'file_name')")
    g.add_argument("--csv-text-col", type=str, default="text",help="CSV column name for text prompt (default 'text')")
    g.add_argument("--video-root", type=str, default=None,help="Root directory containing videos; final path = join(video_root, <csv_video_name>)")

    g = parser.add_argument_group("Sampling")
    g.add_argument("--every-n", type=int, default=None, help="Take 1 frame every N frames")
    g.add_argument("--num-frames", type=int, default=None, help="Uniformly sample K frames")

    g = parser.add_argument_group("Model")
    g.add_argument("--model", type=str, default="ViT-L-14",help="e.g., ViT-L-14, ViT-B-16, ViT-L-14-336")
    g.add_argument("--pretrained", type=str, default="openai",help="e.g., openai, laion2b_s32b_b82k. Ignored if --pretrained-path is set.")
    g.add_argument("--pretrained-path", type=str, default=None,help="Local weights file (.safetensors / .pt / .bin). If set, no network is used.")
    g.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    g.add_argument("--fp16", action="store_true", help="Use FP16 on CUDA")
    g.add_argument("--batch-size", type=int, default=64, help="Frames per CLIP forward pass")

    g = parser.add_argument_group("Output")
    g.add_argument("--out", type=str, default="results.csv", help="Main results CSV")
    g.add_argument("--dump-per-frame", type=str, default=None,help="Optional CSV for per-frame/per-pair values")

    args = parser.parse_args()

    # build (video, prompt) list
    if args.prompts_csv is None:
        assert args.video is not None and args.prompt is not None, \
            "Either provide --prompts-csv, or both --video and --prompt."
        pairs = [(args.video, args.prompt)]
    else:
        pairs = load_pairs_from_csv(args.prompts_csv, args.csv_video_col, args.csv_text_col, args.video_root)

    # load model (offline if local path provided)
    if args.pretrained_path:
        model, tokenizer, preprocess = load_clip_offline(
            args.model, args.pretrained_path, device=args.device, fp16=args.fp16
        )
    else:
        model, tokenizer, preprocess = load_clip_online(
            args.model, args.pretrained, device=args.device, fp16=args.fp16
        )

    rows, per_rows = [], []
    for (vid, txt) in tqdm(pairs, desc="Evaluating"):
        if not os.path.isfile(vid):
            rows.append({"video": vid, "prompt": txt, "num_frames": 0,
                         "clip_t": "", "clip_f": "", "note": "video_not_found"})
            continue

        try:
            frames = read_video_frames(vid, args.every_n, args.num_frames)
        except Exception as e:
            rows.append({"video": vid, "prompt": txt, "num_frames": 0,
                         "clip_t": "", "clip_f": "", "note": f"decode_error:{e}"})
            continue

        clip_t, clip_f, per_t, per_f = clip_T_and_F(
            model, tokenizer, preprocess, frames, txt,
            device=args.device, fp16=args.fp16, batch_size=args.batch_size
        )

        rows.append({
            "video": vid, "prompt": txt.strip()[:50], "num_frames": len(frames),
            "clip_t": float(clip_t) if clip_t == clip_t else "",
            "clip_f": float(clip_f) if clip_f == clip_f else "", "note": ""
        })

        if args.dump_per_frame is not None and len(frames) > 0:
            for i, s in enumerate(per_t.tolist()):
                per_rows.append({"video": vid, "prompt": txt, "type": "CLIP-T", "index": i, "value": float(s)})
            for i, s in enumerate(per_f.tolist()):
                per_rows.append({"video": vid, "prompt": txt, "type": "CLIP-F_pair", "index": i, "value": float(s)})

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)

    if args.dump_per_frame is not None and len(per_rows) > 0:
        os.makedirs(os.path.dirname(args.dump_per_frame) or ".", exist_ok=True)
        pd.DataFrame(per_rows).to_csv(args.dump_per_frame, index=False)

    print(f"Saved summary to {args.out}")
    if args.dump_per_frame:
        print(f"Saved per-frame details to {args.dump_per_frame}")

if __name__ == "__main__":
    main()
