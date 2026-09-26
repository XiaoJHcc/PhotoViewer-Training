"""用训练事件交叉验证选择正则化视觉排序器，留出标签不参与选择。"""
from pathlib import Path
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
import argparse
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'audit'), str(ROOT / 'train')]
from evaluation_protocol import read_rows, write_json, digest, budget_metrics, local_metrics, abs_metrics
from split_eval import summarize_budget
from protocol_eval import joined_labels


def load_data(m3: Path, absolute: Path, snapshot: Path):
    """连接已有监督及照片身份，保留偏好方向、来源与事件。"""
    photo_rows = read_rows(m3 / 'photos.csv')
    metadata = {row['fingerprint']: row for row in photo_rows}
    if len(metadata) != len(photo_rows):
        raise ValueError('照片清单有重复指纹')
    if any(row['split'] not in ('train', 'val', 'test') or int(row['rating_raw']) not in range(6)
           for row in photo_rows):
        raise ValueError('非法划分或原始星级')
    labels = []
    for batch in (2, 3):
        labels.extend(joined_labels(snapshot / f'batch{batch}_key.csv', snapshot / f'batch{batch}.tsv'))
    pairs = []
    for row in read_rows(m3 / 'pairs_train.csv'):
        if row['ptype'] == 'derived':
            continue
        left, right = row['fp_i'], row['fp_j']
        if row['ptype'] not in ('window', 'global') or metadata[left]['event'] != metadata[right]['event']:
            raise ValueError('旧星配对类型非法或跨事件')
        if metadata[left]['rating_raw'] == metadata[right]['rating_raw']:
            raise ValueError('旧星配对无方向')
        if int(metadata[left]['rating_raw']) < int(metadata[right]['rating_raw']):
            left, right = right, left
        pairs.append((left, right, float(row['weight']), row['ptype']))
    abs_rows = read_rows(absolute / 'pairs_train.csv')
    for row in abs_rows:
        pairs.append((row['fp_i'], row['fp_j'], float(row['weight']), 'abs'))
    for row in read_rows(snapshot / 'golden_pairs.csv'):
        if all(metadata[row[field]]['split'] == 'train' for field in ('fp_i', 'fp_j')):
            pairs.append((row['fp_i'], row['fp_j'], float(row['weight']), 'golden'))
    if any(metadata[fingerprint]['split'] != 'train' for row in pairs for fingerprint in row[:2]):
        raise ValueError('训练来源越界')
    if any(row[0] == row[1] or not np.isfinite(row[2]) or row[2] <= 0 for row in pairs):
        raise ValueError('配对自环或权重非法')
    return metadata, labels, pairs, abs_rows


def fit(features, fingerprints, pairs, metadata, allowed_events, penalty):
    """按来源与事件等权拟合成对 logistic；仅拟合允许的训练事件。"""
    index_of = {fingerprint: index for index, fingerprint in enumerate(fingerprints)}
    selected = [row for row in pairs if all(metadata[fp]['event'] in allowed_events for fp in row[:2])]
    directions = defaultdict(set)
    for winner, loser, _, _ in selected:
        directions[tuple(sorted((winner, loser)))].add(winner)
    selected = [row for row in selected if len(directions[tuple(sorted(row[:2]))]) == 1]
    unique = {}
    for winner, loser, weight, source in selected:
        identity = (winner, loser, source)
        if identity not in unique:
            unique[identity] = weight
        else:
            unique[identity] = min(unique[identity], weight)
    selected = [(winner, loser, weight, source) for (winner, loser, source), weight in unique.items()]
    source_shares = {'window': .5, 'global': .25, 'abs': .2, 'golden': .05}
    domains = [(row[3], tuple(sorted(metadata[fp]['event'] for fp in row[:2]))) for row in selected]
    domain_mass = Counter()
    for domain, row in zip(domains, selected):
        domain_mass[domain] += row[2]
    source_domains = Counter(domain[0] for domain in domain_mass)
    weights = np.array([row[2] / domain_mass[domain] * source_shares[row[3]] / source_domains[row[3]]
                        for row, domain in zip(selected, domains)], dtype=np.float64)
    weights /= weights.sum()
    winners = np.array([index_of[row[0]] for row in selected])
    losers = np.array([index_of[row[1]] for row in selected])

    def objective(coefficient):
        """返回逐照片聚合的成对损失及梯度，避免复制全部成对特征。"""
        scores = features @ coefficient
        differences = scores[winners] - scores[losers]
        value = np.dot(weights, np.logaddexp(0, -differences)) + penalty * np.dot(coefficient, coefficient) / 2
        gradient_pair = -weights * expit(-differences)
        gradient_photo = np.bincount(winners, weights=gradient_pair, minlength=len(features))
        gradient_photo -= np.bincount(losers, weights=gradient_pair, minlength=len(features))
        return value, features.T @ gradient_photo + penalty * coefficient

    result = minimize(objective, np.zeros(features.shape[1]), jac=True, method='L-BFGS-B',
                      options={'maxiter': 500, 'ftol': 1e-11, 'gtol': 1e-7})
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError(f'优化未收敛: {result.message}')
    return result.x, {'pairs': len(selected), 'sources': dict(Counter(row[3] for row in selected)),
                      'loss': float(result.fun), 'iterations': result.nit}


def measure(metadata, labels, absolute, scores, events):
    """只在显式事件集合评估局部盲评、直接横评和固定预算。"""
    local = [dict(row, event=metadata[row['fingerprint']]['event']) for row in labels
             if metadata[row['fingerprint']]['event'] in events]
    rows = [row for row in absolute if all(metadata[row[field]]['event'] in events for field in ('fp_i', 'fp_j'))]
    return {'local': local_metrics(local, scores), 'abs': abs_metrics(rows, metadata, scores),
            'budget': summarize_budget(budget_metrics(metadata, scores, sorted(events)))}


def balanced_utility(candidate):
    """按事件宏平均同时衡量减量、局部选优和跨事件排序，避免只凭微小代理差异挑模型。"""
    budget = np.mean([fold['budget']['0.125']['photo_recall_macro'] for fold in candidate['folds']])
    local = np.mean([fold['local']['unique_best']['best1_event_macro'] for fold in candidate['folds']])
    absolute = np.mean([fold['abs']['cross_event']['event_macro'] for fold in candidate['folds']])
    return float((budget * local * absolute) ** (1 / 3))


def main():
    """先在 v2 train 的三折事件留出选择特征与正则，随后仅考 v2 val。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--m3', type=Path, required=True)
    parser.add_argument('--abs-pairs', type=Path, required=True)
    parser.add_argument('--labels-snapshot', type=Path, required=True)
    parser.add_argument('--reuse-folds', type=Path)
    args = parser.parse_args()
    threadpool_limits(2)
    destination = args.output
    destination.mkdir(exist_ok=False)
    metadata, labels, pairs, absolute = load_data(args.m3, args.abs_pairs, args.labels_snapshot)
    cached = np.load(args.features)
    fingerprints = cached['fingerprints'].tolist()
    dino = cached['dino']
    if fingerprints != sorted(metadata) or dino.shape != (len(fingerprints), 384) or cached['features'].shape != (len(fingerprints), 768):
        raise ValueError('特征覆盖、顺序或维度不一致')
    if not np.isfinite(dino).all() or not np.isfinite(cached['features']).all():
        raise ValueError('特征包含非有限值')
    if not all(np.allclose(np.linalg.norm(array, axis=1), 1, atol=1e-4)
               for array in (dino, cached['features'])):
        raise ValueError('视觉特征必须逐张 L2 归一化')
    train_events = sorted({row['event'] for row in metadata.values() if row['split'] == 'train'},
                          key=lambda event: hashlib.sha256(('20260926:' + event).encode()).hexdigest())
    features = {'dino': dino.astype(np.float64) * np.sqrt(dino.shape[1]),
                'clip': cached['features'].astype(np.float64) * np.sqrt(768)}
    features['joint'] = np.concatenate((features['dino'], features['clip']), axis=1) / np.sqrt(2)
    plan = {'folds': [train_events[fold::3] for fold in range(3)], 'penalties': [.01, .1, 1.0, 10.0],
            'source_shares': {'window': .5, 'global': .25, 'abs': .2, 'golden': .05},
            'selection': '三折事件宏平均：照片召回、人工局部冠军、横评事件对准确率三者几何平均；不用 val/test 选参数',
            'script_sha256': digest(Path(__file__)), 'feature_sha256': digest(args.features),
            'input_sha256': {str(path): digest(path) for path in [args.m3 / 'photos.csv', args.m3 / 'pairs_train.csv',
                args.abs_pairs / 'pairs_train.csv', args.labels_snapshot / 'golden_pairs.csv',
                *[args.labels_snapshot / f'batch{batch}{suffix}' for batch in (2, 3) for suffix in ('_key.csv', '.tsv')]]}}
    write_json(destination / 'plan.json', plan)
    if args.reuse_folds:
        previous = json.loads((args.reuse_folds / 'plan.json').read_text(encoding='utf-8'))
        if any(previous.get(field) != plan[field] for field in
               ('feature_sha256', 'folds', 'source_shares', 'penalties', 'input_sha256', 'selection')):
            raise ValueError('不能复用不同特征、监督、抽样或事件划分的交叉验证')
    results = []
    for name, feature in features.items():
        for penalty in plan['penalties']:
            if args.reuse_folds:
                record = json.loads((args.reuse_folds / f'{name}_{penalty}.json').read_text(encoding='utf-8'))
                results.append(record)
                write_json(destination / f'{name}_{penalty}.json', record)
                continue
            folds = []
            for fold, held_out in enumerate(plan['folds']):
                coefficient, trace = fit(feature, fingerprints, pairs, metadata, set(train_events) - set(held_out), penalty)
                scores = dict(zip(fingerprints, (feature @ coefficient).tolist()))
                metrics = measure(metadata, labels, absolute, scores, set(held_out))
                folds.append(metrics)
            budget = float(np.mean([metrics['budget']['0.125']['photo_recall_macro'] for metrics in folds]))
            local = float(np.mean([metrics['local']['unique_best']['best1'] for metrics in folds]))
            cross = float(np.mean([metrics['abs']['cross_event']['accuracy'] for metrics in folds]))
            record = {'features': name, 'penalty': penalty, 'photo_recall': budget,
                      'local_best1': local, 'cross_accuracy': cross, 'folds': folds}
            results.append(record)
            write_json(destination / f'{name}_{penalty}.json', record)
            print(f'{name} penalty={penalty} oof recall={budget:.4f} local={local:.4f} cross={cross:.4f}', flush=True)
    chosen = max(results, key=balanced_utility)
    write_json(destination / 'chosen.json', {key: value for key, value in chosen.items() if key != 'folds'})
    for name in features:
        candidate = max((row for row in results if row['features'] == name), key=balanced_utility)
        coefficient, trace = fit(features[name], fingerprints, pairs, metadata, set(train_events), candidate['penalty'])
        scores = dict(zip(fingerprints, (features[name] @ coefficient).tolist()))
        with (destination / f'{name}_scores.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['fingerprint', 'score'])
            writer.writerows(scores.items())
        np.save(destination / f'{name}_coefficient.npy', coefficient)
        val_events = {row['event'] for row in metadata.values() if row['split'] == 'val'}
        val_abs = read_rows(args.abs_pairs / 'pairs_val.csv')
        validation = measure(metadata, labels, val_abs, scores, val_events)
        write_json(destination / f'{name}_validation.json', {'trace': trace, 'penalty': candidate['penalty'], **validation})
        print(f"{name} val recall={validation['budget']['0.125']['photo_recall_macro']:.4f} "
              f"local={validation['local']['unique_best']['best1']:.4f} cross={validation['abs']['cross_event']['accuracy']:.4f}", flush=True)


if __name__ == '__main__':
    main()
