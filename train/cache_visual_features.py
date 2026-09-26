"""并行消费独立修复缓存，提取冻结 DINO/CLIP，不读取任何评分标签。"""
import csv
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image
import torch
from transformers import AutoModel, CLIPVisionModelWithProjection, CLIPImageProcessorPil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'train'))
sys.path.insert(0, str(ROOT / 'audit'))
from evaluation_protocol import digest


def main():
    """等待每张修复图写完后编码，保存全覆盖特征和图像哈希清单。"""
    torch.set_num_threads(3)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--photos', type=Path, required=True)
    parser.add_argument('--render-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dino-model', type=Path, required=True)
    parser.add_argument('--clip-model', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    metadata_path = args.photos
    with metadata_path.open(encoding='utf-8-sig') as stream:
        fingerprints = sorted(row['fingerprint'] for row in csv.DictReader(stream))
    if not fingerprints or len(set(fingerprints)) != len(fingerprints):
        raise ValueError('照片清单为空或指纹重复')
    index_of = {fp: index for index, fp in enumerate(fingerprints)}
    source = args.render_dir
    if (source / 'decode-version.txt').read_text().strip() != 'display-plane-v3':
        raise ValueError('解码版本错误')
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    clip_path = args.clip_model
    clip = CLIPVisionModelWithProjection.from_pretrained(clip_path, local_files_only=True).to(args.device).eval()
    processor = CLIPImageProcessorPil.from_pretrained(clip_path, local_files_only=True)
    dino = AutoModel.from_pretrained(args.dino_model, local_files_only=True).to(args.device).eval()
    checkpoint = output.with_suffix('.checkpoint')
    checkpoint.mkdir(exist_ok=True)
    identity = {'fingerprints': fingerprints, 'metadata_sha256': hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
                'decode_version': 'display-plane-v3', 'clip_model': str(args.clip_model.resolve()),
                'dino_model': str(args.dino_model.resolve()), 'device': args.device}
    model_assets = [path for model_dir in (args.clip_model, args.dino_model) for path in model_dir.rglob('*')
                    if path.is_file() and path.suffix in ('.safetensors', '.bin', '.json')]
    if not model_assets:
        raise ValueError('模型目录没有可校验的权重或配置')
    identity['model_sha256'] = {str(path.resolve()): digest(path) for path in sorted(model_assets)}
    identity_path = checkpoint / 'identity.json'
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding='utf-8')) != identity:
            raise ValueError('续跑来源或模型身份不一致')
    elif any(checkpoint.iterdir()):
        raise ValueError('已有续跑文件缺少身份清单')
    else:
        identity_path.write_text(json.dumps(identity, ensure_ascii=False), encoding='utf-8')
    arrays = []
    for name, shape in (('clip', (len(fingerprints), 768)), ('dino', (len(fingerprints), 384))):
        array_path = checkpoint / f'{name}.npy'
        if array_path.exists():
            array = np.load(array_path, mmap_mode='r+')
            if array.shape != shape or array.dtype != np.float32:
                raise ValueError(f'续跑特征尺寸错误: {name}')
        else:
            array = np.lib.format.open_memmap(array_path, mode='w+', dtype=np.float32, shape=shape)
            array[:] = np.nan
            array.flush()
        arrays.append(array)
    clip_features, dino_features = arrays
    checkpoint_path = checkpoint / 'progress.json'
    hashes = json.loads(checkpoint_path.read_text(encoding='utf-8')) if checkpoint_path.exists() else {}
    if set(hashes) - set(fingerprints):
        raise ValueError('续跑清单存在未知指纹')
    for fingerprint, expected_hash in hashes.items():
        if hashlib.sha256((source / f'{fingerprint}.png').read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f'续跑图像发生变化: {fingerprint}')
    remaining = set(fingerprints) - hashes.keys()
    deadline = time.monotonic() + 6 * 3600
    mean = torch.tensor([.485, .456, .406], device=args.device).view(1, 3, 1, 1)
    std = torch.tensor([.229, .224, .225], device=args.device).view(1, 3, 1, 1)
    last_report = time.monotonic()
    with torch.inference_mode():
        while remaining:
            if time.monotonic() > deadline:
                raise TimeoutError('六小时内未完成特征提取；进度已落盘')
            ready = sorted(entry.name[:-4] for entry in os.scandir(source)
                           if entry.name.endswith('.png') and entry.name[:-4] in remaining
                           and time.time() - entry.stat().st_mtime > 3)
            for start in range(0, len(ready), 32):
                batch = ready[start:start + 32]
                images = []
                for fp in batch:
                    path = source / f'{fp}.png'
                    with Image.open(path) as image:
                        if image.size != (518, 518):
                            raise ValueError(f'错误图片尺寸 {fp}: {image.size}')
                        images.append(image.convert('RGB'))
                    hashes[fp] = hashlib.sha256(path.read_bytes()).hexdigest()
                pixels = torch.from_numpy(np.stack([np.asarray(image) for image in images])).to(args.device)
                pixels = (pixels.permute(0, 3, 1, 2).float() / 255 - mean) / std
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.device == 'cuda'):
                    dino_vectors = dino(pixel_values=pixels).last_hidden_state[:, 0].float()
                    clip_vectors = clip(pixel_values=processor(images=images, return_tensors='pt').pixel_values.to(args.device)).image_embeds.float()
                dino_vectors = torch.nn.functional.normalize(dino_vectors, dim=-1)
                clip_vectors = torch.nn.functional.normalize(clip_vectors, dim=-1)
                indexes = [index_of[fp] for fp in batch]
                dino_features[indexes] = dino_vectors.cpu().numpy()
                clip_features[indexes] = clip_vectors.cpu().numpy()
                for array in arrays:
                    array.flush()
                temporary = checkpoint_path.with_suffix('.tmp')
                temporary.write_text(json.dumps(hashes), encoding='utf-8')
                temporary.replace(checkpoint_path)
                remaining.difference_update(batch)
            if time.monotonic() - last_report > 45:
                print(f'corrected features {len(fingerprints)-len(remaining)}/{len(fingerprints)}', flush=True)
                last_report = time.monotonic()
            if remaining and (source / 'manifest.json').exists():
                manifest = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
                if manifest['Failures']:
                    raise ValueError(f"上游修复失败: {len(manifest['Failures'])}")
                absent = {fp for fp in remaining if not (source / f'{fp}.png').exists()}
                if absent:
                    raise ValueError(f'上游完成但仍缺图 {len(absent)}')
            if remaining:
                time.sleep(3)
    if not all(np.isfinite(array).all() for array in (dino_features, clip_features)):
        raise ValueError('全库特征不完整')
    temporary_output = output.with_suffix('.partial.npz')
    np.savez(temporary_output, fingerprints=np.array(fingerprints), dino=dino_features, features=clip_features)
    temporary_output.replace(output)
    output.with_suffix('.json').write_text(json.dumps({'decode_version': 'display-plane-v3',
        'model_identity': identity, 'images': hashes, 'metadata_sha256': hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2), encoding='utf-8')
    print('corrected feature cache complete', flush=True)


if __name__ == '__main__':
    main()
