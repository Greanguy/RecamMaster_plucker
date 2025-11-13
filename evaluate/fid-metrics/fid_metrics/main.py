import hydra
from hydra.core.hydra_config import HydraConfig
import os
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from rich.progress import track
from pathlib import Path
from collections import defaultdict 

from fid_metrics import (
    ImageDataset,
    ImageSequenceDataset,
    VideoDataset,
    build_inception,
    build_inception3d,
    calculate_fid,
    is_image_dir_path,
    is_video_path,
    postprocess_i2d_pred,
)

def build_loaders(type, paths, cfg):
    dls = []
    for path in paths:
        bs = cfg.batch_size
        dataset_cfgs = cfg.get('dataset')

        if is_video_path(path):
            if type == 'fid':
                if dataset_cfgs:
                    dataset_cfgs = dict(dataset_cfgs)
                    dataset_cfgs['sequence_length'] = bs
                else:
                    dataset_cfgs = {'sequence_length': bs}
                bs = 1
            C = VideoDataset
        elif is_image_dir_path(path):
            C = ImageDataset if type == 'fid' else ImageSequenceDataset
        else:
            raise NotImplementedError

        dataset = C(path, **dataset_cfgs) if dataset_cfgs else C(path)
        dl = torch.utils.data.DataLoader(dataset, bs, shuffle=True, num_workers=cfg.num_workers)
        dls.append(dl)
    return dls

def build_model(type, cfg):
    if type == 'fid':
        return build_inception(cfg.dims)
    elif type in ('fvd', 'fvd_v'): 
        return build_inception3d(cfg.type, cfg.path)
    else:
        raise NotImplementedError

# ========= 枚举子目录并配对 =========
def get_video_pairs(real_root: Path, gen_root: Path):
    """返回 [(real_dir, gen_dir, video_name), ...]"""
    real_root = Path(real_root)
    gen_root = Path(gen_root)

    real_dirs = sorted([p for p in real_root.iterdir() if p.is_dir()])
    gen_dirs = {p.name: p for p in gen_root.iterdir() if p.is_dir()}

    pairs = []
    for r in real_dirs:
        name = r.name
        if name in gen_dirs:
            pairs.append((r, gen_dirs[name], name))
        else:
            print(f"[WARN] 没找到生成视频子目录: {name}，跳过该视频")
    return pairs

# ========= 多视角 FVD-V: 枚举 video / timestep =========
def get_view_timestep_pairs(real_root: Path, gen_root: Path):
    """
    返回 [(real_t_dir, gen_t_dir, video_name, timestep_name), ...]
    其中：
      real_t_dir:  view_frames_real/videoXXX/tYYYY
      gen_t_dir:   view_frames_gen/videoXXX/tYYYY
    """
    real_root = Path(real_root)
    gen_root = Path(gen_root)

    pairs = []

    # 枚举所有 video 目录
    for real_video_dir in sorted(real_root.iterdir()):
        if not real_video_dir.is_dir():
            continue
        video_name = real_video_dir.name
        gen_video_dir = gen_root / video_name
        if not gen_video_dir.is_dir():
            print(f"[WARN] 生成目录缺少视频 {video_name}，跳过该视频")
            continue

        # 该 video 下的所有 timestep 子目录
        real_ts_dirs = sorted([p for p in real_video_dir.iterdir() if p.is_dir()])
        gen_ts_map = {p.name: p for p in gen_video_dir.iterdir() if p.is_dir()}

        for rt in real_ts_dirs:
            tname = rt.name
            if tname not in gen_ts_map:
                print(f"[WARN] 视频 {video_name} 缺少 timestep {tname} 的生成结果，跳过该 timestep")
                continue
            pairs.append((rt, gen_ts_map[tname], video_name, tname))

    return pairs


# ========= 从一对路径中抽取特征 =========
def extract_features_for_pair(metric_type, metric_data_cfg, model, device, paths, num_iters=None):
    """
    paths: [real_path, gen_path]
    返回：feats_real, feats_fake (numpy.array)
    """
    dls = build_loaders(metric_type, paths, metric_data_cfg)

    feats = [[], []]
    for i, dl in enumerate(dls):
        # num_iters 优先，否则跑完整个 dataloader
        if num_iters is not None:
            # 防止 num_iters 太大导致 StopIteration，这里做个保护
            max_iters = max(1, num_iters // metric_data_cfg.batch_size)
            seq = range(max_iters)
        else:
            seq = range(len(dl))
        dl_iter = iter(dl)

        for _ in seq:
            try:
                x = next(dl_iter).to(device)
            except StopIteration:
                break

            if metric_type == 'fid' and x.dim() == 5:
                # [B, T, C, H, W] -> [B*T, C, H, W]
                x = x.squeeze(0).transpose(0, 1)
            elif metric_type == 'fvd':
                # 映射到 [-1,1]
                x = x * 2 - 1

            with torch.no_grad():
                if metric_type == 'fid':
                    pred = model(x)
                    pred = postprocess_i2d_pred(pred)
                elif metric_type == 'fvd':
                    # styleganv 
                    # 注意：这里 metric_data_cfg.model 在 main 里传入的是 metric_cfgs.model
                    # pred = model(x, return_features=True) if metric_data_cfg.model.type == 'styleganv' else model(x)
                    pred = model(x)
                else:
                    raise NotImplementedError

            feats[i].append(pred.cpu().numpy())

        if len(feats[i]) == 0:
            raise RuntimeError(f"路径 {paths[i]} 没有得到任何特征，请检查数据。")
        feats[i] = np.concatenate(feats[i], axis=0)

    return feats[0], feats[1]


@hydra.main(config_path='../configs', config_name='config', version_base=None)
def main(cfg: DictConfig):    
    print(OmegaConf.to_yaml(cfg))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # cfg.paths 现在假定为:
    #   paths:
    #     - /path/to/real_videos
    #     - /path/to/generate_videos
    assert len(cfg.paths) == 2, "现在的批量模式假定 cfg.paths 里是 [real_root, gen_root]"
    real_root, gen_root = map(Path, cfg.paths)

    hydra_run_dir = Path(HydraConfig.get().runtime.output_dir)
    save_path = hydra_run_dir / "results.txt"

    for metric_cfgs in cfg.metrics:
        metric_type = metric_cfgs.type  # 'fid'  'fvd' 'fvd-v'
        model = build_model(metric_type, metric_cfgs.model).to(device).eval()# 预先 build 一次模型，所有视频共用

        # ===================== FVD-V 分支 =====================
        if metric_type == 'fvd_v':
            print("Running FVD-V (multi-view, timestep-aware)...")
            vt_pairs = get_view_timestep_pairs(real_root, gen_root) # 枚举所有 (video, timestep)
            if len(vt_pairs) == 0:
                print("[WARN] 没有找到任何 (video, timestep) 配对，检查目录结构是否为 videoXXX/tYYYY/frame_*.jpg")
                continue

            # 按 video 聚合特征：video_name -> [feat_array_1, feat_array_2, ...]
            video_feats_real = defaultdict(list)
            video_feats_fake = defaultdict(list)

            # 全局 pooled 特征列表（可选）
            all_feats_real = []
            all_feats_fake = []

            for real_t_dir, gen_t_dir, vname, tname in track(
                vt_pairs, description='FVD-V feature extraction per (video, timestep)'
            ):
                # 注意：这里 metric_type 仍然填 'fvd',因为 extract_features_for_pair内部只认识 'fid' / 'fvd' 两种分支
                feats_real, feats_fake = extract_features_for_pair(
                    metric_type='fvd',
                    metric_data_cfg=metric_cfgs.data,
                    model=model,
                    device=device,
                    paths=[str(real_t_dir), str(gen_t_dir)],
                    num_iters=cfg.get('num_iters')
                )
                # 该 video 下的所有 timestep 特征都 append 上去
                video_feats_real[vname].append(feats_real)
                video_feats_fake[vname].append(feats_fake)

                # 同时加入全局列表
                all_feats_real.append(feats_real)
                all_feats_fake.append(feats_fake)

            # 1) 计算每个 video 的 FVD-V
            per_video_scores = []
            for vname in sorted(video_feats_real.keys()):
                fr = np.concatenate(video_feats_real[vname], axis=0)  # 所有 timestep 的特征拼在一起
                ff = np.concatenate(video_feats_fake[vname], axis=0)

                score = calculate_fid(fr, ff)  # 用 I3D 特征算 FID，即 FVD-V
                per_video_scores.append(score)

                print(f'FVD-V [{vname}]: {score:.6f}')
                with open(save_path, "a") as f:
                    f.write(f"FVD-V [{vname}]: {score:.6f}\n")

            if len(per_video_scores) > 0:
                # 2) 所有 video 的平均 FVD-V
                mean_score = float(np.mean(per_video_scores))
                print(f'FVD-V MEAN over {len(per_video_scores)} videos (per-video scores): {mean_score:.6f}')
                with open(save_path, "a") as f:
                    f.write(f"FVD-V MEAN over {len(per_video_scores)} videos (per-video scores): {mean_score:.6f}\n")

                # 3) 全局 pooled FVD-V（如果你只想要 2) 可以把这段删掉）
                all_fr = np.concatenate(all_feats_real, axis=0)
                all_ff = np.concatenate(all_feats_fake, axis=0)
                global_score = calculate_fid(all_fr, all_ff)
                print(f'FVD-V GLOBAL (all videos, all timesteps pooled): {global_score:.6f}')
                with open(save_path, "a") as f:
                    f.write(f"FVD-V GLOBAL (all videos, all timesteps pooled): {global_score:.6f}\n")
            else:
                print("[WARN] 没有有效的视频来计算 FVD-V")

            # 这一轮 metric 已经处理完了 继续下一个 metric_cfgs
            continue
        
        # ===================== FVD/FID  =====================
        pairs = get_video_pairs(real_root, gen_root)# 获取 (真实视频子目录, 生成视频子目录, 名字) 列表

        all_scores = []

        for real_dir, gen_dir, name in track(pairs, description=f'{metric_type} per video'):
            # 对每一对视频子目录提特征、算 FID/FVD
            feats_real, feats_fake = extract_features_for_pair(
                metric_type,
                metric_cfgs.data,
                model,
                device,
                [str(real_dir), str(gen_dir)],
                num_iters=cfg.get('num_iters')
            )

            score = calculate_fid(feats_real, feats_fake)
            all_scores.append(score)

            print(f'{metric_type.upper()} [{name}]: {score:.6f}')

            # 持续写入到文件里
            with open(save_path, "a") as f:
                f.write(f"{metric_type.upper()} [{name}]: {score:.6f}\n")

        # 所有视频的平均分数
        if len(all_scores) > 0:
            mean_score = float(np.mean(all_scores))
            print(f'{metric_type.upper()} MEAN over {len(all_scores)} videos: {mean_score:.6f}')
            with open(save_path, "a") as f:
                f.write(f"{metric_type.upper()} MEAN over {len(all_scores)} videos: {mean_score:.6f}\n")
        else:
            print(f"[WARN] 没有有效的视频对来计算 {metric_type.upper()}")


if __name__ == '__main__':
    main()
