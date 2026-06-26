#!/usr/bin/env python3
import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / 'results/paper/figures'
CACHE = ROOT / 'results/paper/full_retrain_sameframe/dataset_cache/dataset_sameframe_pose_30k'
FINAL = ROOT / 'results/paper/final_validation'
FIG_DIR.mkdir(parents=True, exist_ok=True)

PVE_MEAN = 1.947495
LBS_BASELINE = 5.419208
IMPROVEMENT = 64.1
GPU_LAT_MS = 1.094395
ONNX_CPU = 5.256
ONNX_NNAPI = 5.277
DQS_BASELINE = 1.278
FP16_MAX = 4.3869e-05
DEVICE = 'TECNO KL4 / Mali-G57 / Android 14'

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.25,
})

manifest = []

def require(path):
    path = ROOT / path if not isinstance(path, Path) else path
    if not path.exists():
        raise FileNotFoundError(str(path))
    return path

def save_fig(fig, stem):
    pdf = FIG_DIR / f'{stem}.pdf'
    png = FIG_DIR / f'{stem}.png'
    fig.tight_layout(pad=0.6)
    fig.savefig(pdf, bbox_inches='tight')
    fig.savefig(png, bbox_inches='tight', dpi=300)
    plt.close(fig)


def as_memmap(path, shape=None, dtype=None):
    arr = np.load(path, mmap_mode='r')
    return arr


def load_train_log():
    rows = []
    for line in require('results/paper/full_retrain_sameframe/train.log').read_text().splitlines():
        line = line.strip()
        if not line.startswith('{'):
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if 'epoch' in obj and 'val_pve_mm' in obj:
            rows.append(obj)
    if not rows:
        raise RuntimeError('No JSON epoch rows with val_pve_mm in train.log')
    return rows


def load_pve():
    values = []
    with require('results/paper/final_validation/full_30k_val_pve_per_pose.csv').open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            values.append(float(row['per_pose_pve_mm']))
    return np.asarray(values, dtype=np.float32)


def rot6d_to_angle(rot6d):
    # rot6d shape (..., 24, 6), first two columns of rotation matrix.
    a1 = rot6d[..., 0:3]
    a2 = rot6d[..., 3:6]
    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-8)
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / np.maximum(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-8)
    b3 = np.cross(b1, b2)
    # columns b1,b2,b3; trace is sum diagonal entries.
    trace = b1[..., 0] + b2[..., 1] + b3[..., 2]
    cos = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return np.arccos(cos)


def max_joint_angles(indices=None):
    theta = as_memmap(require(CACHE / 'theta_rot6d.npy'))
    if indices is not None:
        theta = theta[indices]
    n = theta.shape[0]
    out = np.empty(n, dtype=np.float32)
    chunk = 512
    for s in range(0, n, chunk):
        x = np.asarray(theta[s:s+chunk], dtype=np.float32).reshape(-1, 24, 6)
        out[s:s+chunk] = rot6d_to_angle(x).max(axis=1)
    return out


def target_mean_magnitude(indices=None):
    target = as_memmap(require(CACHE / 'target_delta_mm.npy'))
    if indices is None:
        n = target.shape[0]
        get = lambda s,e: target[s:e]
    else:
        n = len(indices)
        get = lambda s,e: target[indices[s:e]]
    out = np.empty(n, dtype=np.float32)
    chunk = 128
    for s in range(0, n, chunk):
        arr = np.asarray(get(s, min(n, s+chunk)), dtype=np.float32).reshape(-1, 6890, 3)
        out[s:min(n, s+chunk)] = np.linalg.norm(arr, axis=2).mean(axis=1)
    return out


def parse_weights_fp16():
    raw = np.fromfile(require('models/weights_sameframe_fp16.bin'), dtype=np.float16)
    cursor = 0
    def take(count):
        nonlocal cursor
        x = raw[cursor:cursor+count].astype(np.float32)
        cursor += count
        return x
    W0 = take(64*144).reshape(64,144); b0 = take(64)
    W1 = take(64*64).reshape(64,64); b1 = take(64)
    W2 = take(20670*64).reshape(20670,64); b2 = take(20670)
    if cursor != raw.size:
        raise RuntimeError(f'weight parse mismatch cursor={cursor} size={raw.size}')
    return W0,b0,W1,b1,W2,b2


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x*x*x)))


def infer_batches(theta, indices, batch=32):
    W0,b0,W1,b1,W2,b2 = parse_weights_fp16()
    n = len(indices)
    for s in range(0, n, batch):
        idx = indices[s:s+batch]
        x = np.asarray(theta[idx], dtype=np.float32)
        h0 = gelu(x @ W0.T + b0)
        h1 = gelu(h0 @ W1.T + b1)
        y = h1 @ W2.T + b2
        yield s, idx, y.astype(np.float32)


def sampled_vertex_errors(sample_count=500):
    theta = as_memmap(require(CACHE / 'theta_rot6d.npy'))
    target = as_memmap(require(CACHE / 'target_delta_mm.npy'))
    val_indices = np.load(require(CACHE / 'val_indices.npy'))
    pve = load_pve()
    rng = np.random.default_rng(20260626)
    # Include easy/medium/hard candidates plus random sample for distribution.
    easy_pos = int(np.argmin(pve))
    medium_pos = int(np.argmin(np.abs(pve - PVE_MEAN)))
    hard_pos = int(np.argmin(np.abs(pve - np.percentile(pve, 90))))
    base_positions = np.array([easy_pos, medium_pos, hard_pos], dtype=np.int64)
    remaining = np.setdiff1d(np.arange(len(val_indices)), base_positions, assume_unique=False)
    extra = rng.choice(remaining, size=max(0, sample_count-len(base_positions)), replace=False)
    positions = np.concatenate([base_positions, np.sort(extra)])
    indices = val_indices[positions]
    vertex_errors = []
    selected = {}
    for s, idx, pred in infer_batches(theta, indices, batch=24):
        tgt = np.asarray(target[idx], dtype=np.float32)
        err = np.linalg.norm((pred - tgt).reshape(-1, 6890, 3), axis=2)
        vertex_errors.append(err.astype(np.float32))
        for local, pos in enumerate(positions[s:s+len(idx)]):
            if pos in base_positions:
                selected[int(pos)] = (pred[local].copy(), tgt[local].copy(), err[local].copy())
    return positions, indices, np.vstack(vertex_errors), selected


def fig_dataset_distribution():
    max_ang = max_joint_angles()
    mean_mag = target_mean_magnitude()
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.6))
    bins = np.linspace(0, 6, 50)
    ax[0].hist(max_ang, bins=bins, density=True, color='#4C78A8', alpha=0.75, label='CMU MoCap')
    syn = np.linspace(0, 6, 30000)
    ax[0].hist(syn, bins=bins, density=True, color='#F585A8', alpha=0.35, label='Synthetic flat ref.')
    ax[0].axvline(0.5, ls='--', c='k', lw=1); ax[0].axvline(2.5, ls='--', c='k', lw=1)
    ax[0].set_xlabel('Max joint rotation per pose (rad)'); ax[0].set_ylabel('Density'); ax[0].legend(frameon=False)
    ax[1].hist(mean_mag, bins=60, color='#4C78A8', alpha=0.8)
    ax[1].axvline(LBS_BASELINE, ls='--', c='#D62728', lw=1.2, label='LBS baseline 5.419mm')
    ax[1].set_xlabel('Mean |δv| per pose (mm)'); ax[1].set_ylabel('Pose count'); ax[1].legend(frameon=False)
    save_fig(fig, 'fig_dataset_distribution')


def fig_convergence():
    rows = load_train_log()
    e = np.array([r['epoch'] for r in rows])
    val_pve = np.array([r['val_pve_mm'] for r in rows])
    val_mse = np.array([r['val_mse'] for r in rows])
    train_mse = np.array([r['train_mse'] for r in rows])
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 2.6))
    ax[0].plot(e, val_pve, color='#4C78A8', lw=1.3)
    ax[0].axhline(LBS_BASELINE, ls='--', c='#D62728', lw=1, label='LBS (5.419 mm)')
    def mark(epoch, y, marker, color, label): ax[0].scatter([epoch], [y], marker=marker, s=38, c=color, zorder=5, label=label)
    mark(1, val_pve[e==1][0], '*', '#2CA02C', 'LBS crossing e1')
    mark(6, val_pve[e==6][0], '^', '#9467BD', 'small-run crossing e6')
    mark(375, val_pve[e==375][0], 'o', '#D62728', 'checkpoint e375')
    ax[0].set_xlabel('Epoch'); ax[0].set_ylabel('Per-vertex error (mm)'); ax[0].legend(frameon=False)
    ax[1].plot(e, val_mse, color='#4C78A8', lw=1.2, label='val MSE')
    ax[1].plot(e, train_mse, color='#F58518', lw=1.2, ls='--', label='train MSE')
    ax[1].axvline(387, c='k', ls='--', lw=1, label='early stop e387')
    ax[1].set_yscale('log'); ax[1].set_xlabel('Epoch'); ax[1].set_ylabel('MSE loss (log)'); ax[1].legend(frameon=False)
    save_fig(fig, 'fig_convergence')


def fig_baseline_comparison():
    pve = load_pve()
    val_indices = np.load(require(CACHE / 'val_indices.npy'))
    lbs = target_mean_magnitude(val_indices)
    diff = lbs - pve
    maxang = max_joint_angles(val_indices)
    fig, ax = plt.subplots(1, 3, figsize=(6.8, 2.6))
    bp = ax[0].boxplot([lbs, pve], labels=['LBS', 'ZD-ECN'], patch_artist=True, showfliers=False)
    for patch, c in zip(bp['boxes'], ['#F58518', '#4C78A8']): patch.set_facecolor(c); patch.set_alpha(0.65)
    ax[0].axhline(DQS_BASELINE, c='#2CA02C', ls='--', lw=1, label='DQS validated separately, 1.278mm')
    ax[0].set_ylabel('PVE (mm)'); ax[0].legend(frameon=False, loc='upper right')
    sc = ax[1].scatter(lbs, pve, c=maxang, s=4, alpha=0.35, cmap='viridis')
    lim = [0, max(float(lbs.max()), float(pve.max()))*1.02]
    ax[1].plot(lim, lim, c='k', ls='--', lw=1); ax[1].set_xlim(lim); ax[1].set_ylim(lim)
    ax[1].set_xlabel('LBS PVE (mm)'); ax[1].set_ylabel('ZD-ECN PVE (mm)')
    fig.colorbar(sc, ax=ax[1], fraction=0.046, pad=0.02, label='max angle')
    ax[2].hist(diff, bins=60, color='#4C78A8', alpha=0.8)
    ax[2].axvline(float(diff.mean()), c='#D62728', ls='--', lw=1, label=f'mean {diff.mean():.2f}mm')
    ax[2].axvline(0, c='k', ls='--', lw=1, label='break-even')
    ax[2].set_xlabel('LBS PVE − ZD-ECN PVE (mm)'); ax[2].set_ylabel('Pose count'); ax[2].legend(frameon=False)
    save_fig(fig, 'fig_baseline_comparison')


def joint_assignment():
    try:
        d = np.load(require('models/smpl_reference_data/lbs_weights_sparse.npz'))
        if {'indices','offsets','values'} <= set(d.files):
            offsets=d['offsets']; indices=d['indices']; values=d['values']
        else:
            raise KeyError(d.files)
    except Exception:
        offsets=np.fromfile(require('models/smpl_reference_data/lbs_weights_offsets_i32.bin'), dtype=np.int32)
        indices=np.fromfile(require('models/smpl_reference_data/lbs_weights_indices_i32.bin'), dtype=np.int32)
        values=np.fromfile(require('models/smpl_reference_data/lbs_weights_values_f32.bin'), dtype=np.float32)
    assign=np.zeros(6890,dtype=np.int32)
    for v in range(6890):
        s,e=offsets[v],offsets[v+1]
        assign[v]=indices[s+int(np.argmax(values[s:e]))]
    return assign


def fig_perjoint_and_error_and_heatmap():
    positions, indices, verr, selected = sampled_vertex_errors(500)
    assign = joint_assignment()
    names = ['root','l_hip','r_hip','spine1','l_knee','r_knee','spine2','l_ankle','r_ankle','spine3','l_foot','r_foot','neck','l_collar','r_collar','head','l_shoulder','r_shoulder','l_elbow','r_elbow','l_wrist','r_wrist','l_hand','r_hand']
    means = np.array([verr[:, assign==j].mean() if np.any(assign==j) else np.nan for j in range(24)])
    order = np.argsort(means)
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    cmap = plt.get_cmap('coolwarm'); norm = Normalize(vmin=np.nanmin(means), vmax=np.nanmax(means))
    ax.barh(np.arange(24), means[order], color=cmap(norm(means[order])))
    ax.set_yticks(np.arange(24)); ax.set_yticklabels([names[i] for i in order], fontsize=5.8)
    ax.axvline(PVE_MEAN, c='k', ls='--', lw=1, label='overall 1.947mm')
    ax.set_xlabel('Mean sampled vertex error (mm)'); ax.legend(frameon=False)
    save_fig(fig, 'fig_perjoint_pve')

    flat = verr.ravel()
    mean, med, p95, p99 = [float(x) for x in (flat.mean(), np.median(flat), np.percentile(flat,95), np.percentile(flat,99))]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.hist(np.clip(flat,0,20), bins=80, color='#4C78A8', alpha=0.85)
    for val,c,lbl in [(mean,'#4C78A8',f'mean {mean:.2f}'),(med,'k',f'median {med:.2f}'),(p95,'#F58518',f'p95 {p95:.2f}'),(p99,'#D62728',f'p99 {p99:.2f}')]:
        ax.axvline(val, c=c, ls='--', lw=1, label=lbl)
    ax.set_xlim(0,20); ax.set_xlabel('Per-vertex error (mm)'); ax.set_ylabel('Vertex samples')
    ax.set_title('Per-vertex error distribution (N≈3.5M sampled vertices)')
    ax.legend(frameon=False)
    save_fig(fig, 'fig_error_distribution')

    v_template = np.fromfile(require('models/smpl_reference_data/v_template_f32.bin'), dtype=np.float32).reshape(6890,3)
    pve = load_pve()
    easy_pos = int(np.argmin(pve)); medium_pos = int(np.argmin(np.abs(pve-PVE_MEAN))); hard_pos = int(np.argmin(np.abs(pve-np.percentile(pve,90))))
    fig, axes = plt.subplots(1,3,figsize=(6.8,2.6))
    for ax, pos, title in zip(axes, [easy_pos, medium_pos, hard_pos], ['Easy','Medium','Hard']):
        pred, tgt, err = selected[pos]
        pts = v_template + pred.reshape(6890,3)/1000.0
        sc=ax.scatter(pts[:,0], pts[:,1], c=np.clip(err,0,10), s=1.2, cmap='plasma', vmin=0, vmax=10)
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f'{title} (PVE={pve[pos]:.2f}mm)')
    fig.colorbar(sc, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02, label='error mm')
    save_fig(fig, 'fig_vertex_heatmap')
    return flat


def fig_pve_vs_difficulty():
    pve = load_pve(); val_indices=np.load(require(CACHE/'val_indices.npy')); maxang=max_joint_angles(val_indices)
    fig, ax=plt.subplots(figsize=(3.4,2.6))
    ax.scatter(maxang,pve,s=4,alpha=0.15,c='#4C78A8',edgecolors='none')
    bins=np.linspace(maxang.min(),maxang.max(),21); centers=(bins[:-1]+bins[1:])/2; means=[]
    for a,b in zip(bins[:-1],bins[1:]):
        m=(maxang>=a)&(maxang<b); means.append(np.nan if not m.any() else pve[m].mean())
    ax.plot(centers,means,c='#D62728',lw=1.5,label='running mean')
    ax.axhline(LBS_BASELINE,c='#D62728',ls='--',lw=1,label='LBS 5.419mm')
    ax.axhline(DQS_BASELINE,c='#2CA02C',ls='--',lw=1,label='DQS 1.278mm')
    ax.set_xlabel('Max joint angle per pose (rad)'); ax.set_ylabel('ZD-ECN PVE (mm)'); ax.legend(frameon=False)
    save_fig(fig,'fig_pve_vs_difficulty')


def fig_latency_cdf():
    vals=[]
    with require('benchmark_results/raw/benchmark_raw.csv').open() as f:
        for row in csv.DictReader(f): vals.append(float(row['gpu_ms']))
    x=np.sort(np.asarray(vals)); y=np.arange(1,len(x)+1)/len(x)
    fig,ax=plt.subplots(figsize=(3.4,2.6)); ax.plot(x,y,c='#4C78A8',lw=1.5)
    for val,c,l in [(GPU_LAT_MS,'#4C78A8','mean 1.094'),(np.percentile(x,95),'#F58518','p95'),(np.percentile(x,99),'#D62728','p99')]: ax.axvline(val,c=c,ls='--',lw=1,label=l)
    ax.set_xlim(1.0,1.6); ax.set_ylim(0,1.01); ax.set_xlabel('GPU elapsed time (ms)'); ax.set_ylabel('CDF'); ax.set_title('Latency CDF — Mali-G57 (N=5000)'); ax.text(0.04,0.12,'All frames within 60Hz budget (16.7ms)',transform=ax.transAxes); ax.legend(frameon=False)
    save_fig(fig,'fig_latency_cdf')


def fig_session_stability():
    by={}
    with require('benchmark_results/raw/benchmark_raw.csv').open() as f:
        for row in csv.DictReader(f): by.setdefault(int(row['session']),[]).append(float(row['gpu_ms']))
    data=[np.asarray(by[i]) for i in sorted(by)]
    grand=np.concatenate(data).mean()
    means=np.array([d.mean() for d in data]); ns=np.array([len(d) for d in data]); k=len(data); n=int(ns[0])
    within=np.mean([d.var(ddof=1) for d in data]); between=n*means.var(ddof=1); icc=(between-within)/(between+(n-1)*within) if (between+(n-1)*within)!=0 else 0
    fig,ax=plt.subplots(figsize=(3.4,2.6)); ax.boxplot(data,labels=[f'S{i}' for i in sorted(by)],showfliers=False)
    ax.axhline(grand,c='#D62728',ls='--',lw=1,label=f'grand mean {grand:.3f}ms')
    ax.text(0.03,0.92,f'ICC={icc:.3f}',transform=ax.transAxes); ax.set_xlabel('Session'); ax.set_ylabel('GPU time (ms)'); ax.legend(frameon=False)
    save_fig(fig,'fig_session_stability')


def fig_roofline():
    intensity=np.logspace(-2,2,300); peak=460; bw=12; roof=np.minimum(peak,intensity*bw)
    fig,ax=plt.subplots(figsize=(3.4,2.6)); ax.loglog(intensity,roof,c='k',lw=1.5,label='Mali-G57 roofline')
    h64_i=2.67e6/(1356990*2); h64_p=2.67e6/(1.094e-3)/1e9
    h256_i=10.75e6/(5396670*4); h256_p=10.75e6/(17.202e-3)/1e9
    ax.scatter([h64_i,h256_i],[h64_p,h256_p],c=['#4C78A8','#D62728'],s=35,zorder=5)
    ax.annotate('h=64 fp16\n1.094ms, 2.9× overhead',(h64_i,h64_p),xytext=(8,10),textcoords='offset points')
    ax.annotate('h=256 fp32\n17.202ms, 9.6× overhead',(h256_i,h256_p),xytext=(8,-22),textcoords='offset points')
    ax.fill_between(intensity, np.maximum(1e-3, roof/20), roof, color='gray', alpha=0.08)
    ax.text(0.04,0.08,'SSBO cache miss overhead',transform=ax.transAxes,color='gray')
    ax.set_xlabel('Arithmetic intensity (FLOP/byte)'); ax.set_ylabel('Performance (GFLOPS)'); ax.set_ylim(0.1,700); ax.legend(frameon=False)
    save_fig(fig,'fig_roofline')


def parse_weights_file(path, dtype):
    raw = np.fromfile(require(path), dtype=dtype).astype(np.float32)
    cursor = 0
    def take(count):
        nonlocal cursor
        x = raw[cursor:cursor+count]
        cursor += count
        return x
    W0 = take(64*144).reshape(64,144); b0 = take(64)
    W1 = take(64*64).reshape(64,64); b1 = take(64)
    W2 = take(20670*64).reshape(20670,64); b2 = take(20670)
    if cursor != raw.size:
        raise RuntimeError(f'weight parse mismatch for {path}: cursor={cursor} size={raw.size}')
    return W0,b0,W1,b1,W2,b2


def infer_with_weights(x, weights):
    W0,b0,W1,b1,W2,b2 = weights
    h0 = gelu(x @ W0.T + b0)
    h1 = gelu(h0 @ W1.T + b1)
    return h1 @ W2.T + b2

def fig_fp16_error():
    summary=json.loads(require('results/paper/final_validation/glsl_validation_summary.json').read_text())
    measured_max=float(summary.get('Validation max_abs',FP16_MAX))
    measured_mean=float(summary.get('Validation mean_abs',2.48385e-06))
    theta = as_memmap(require(CACHE / 'theta_rot6d.npy'))
    val_indices = np.load(require(CACHE / 'val_indices.npy'))[:200]
    w32 = parse_weights_file('models/weights_sameframe_fp32.bin', np.float32)
    w16 = parse_weights_file('models/weights_sameframe_fp16.bin', np.float16)
    errs=[]
    for s in range(0, len(val_indices), 16):
        x=np.asarray(theta[val_indices[s:s+16]], dtype=np.float32)
        y32=infer_with_weights(x, w32)
        y16=infer_with_weights(x, w16)
        err=np.linalg.norm((y32-y16).reshape(-1,6890,3), axis=2)
        errs.append(err.astype(np.float32))
    vals=np.concatenate([e.ravel() for e in errs])
    mean=float(vals.mean()); p99=float(np.percentile(vals,99)); p999=float(np.percentile(vals,99.9)); mx=float(vals.max())
    fig,ax=plt.subplots(figsize=(3.4,2.6)); ax.hist(np.maximum(vals,1e-9),bins=np.logspace(-9, max(-4, np.log10(max(mx,1e-8))), 80),color='#4C78A8',alpha=0.85)
    ax.set_xscale('log')
    for val,c,l in [(mean,'#4C78A8',f'mean {mean:.1e}'),(p99,'#F58518',f'p99 {p99:.1e}'),(p999,'#D62728',f'p99.9 {p999:.1e}'),(measured_max,'k',f'device max {measured_max:.1e}')]: ax.axvline(max(val,1e-9),c=c,ls='--',lw=1,label=l)
    ax.set_xlabel('Quantisation error (mm)'); ax.set_ylabel('Vertex samples'); ax.set_title('fp16 quantisation error (N=1.38M vertices)')
    ax.text(0.03,0.90,f'Device validation mean {measured_mean:.1e}mm\n4 orders below 1mm',transform=ax.transAxes,va='top'); ax.legend(frameon=False)
    save_fig(fig,'fig_fp16_error')


def run(name, fn):
    try:
        fn(); manifest.append((name,'GENERATED',''))
    except Exception as e:
        manifest.append((name,'FAILED',str(e)))


def main():
    run('fig_dataset_distribution', fig_dataset_distribution)
    run('fig_convergence', fig_convergence)
    run('fig_baseline_comparison', fig_baseline_comparison)
    # Generates fig4, fig6, fig7 in one sampled inference pass.
    run('fig_perjoint_pve / fig_error_distribution / fig_vertex_heatmap', fig_perjoint_and_error_and_heatmap)
    run('fig_pve_vs_difficulty', fig_pve_vs_difficulty)
    run('fig_latency_cdf', fig_latency_cdf)
    run('fig_session_stability', fig_session_stability)
    run('fig_roofline', fig_roofline)
    run('fig_fp16_error', fig_fp16_error)
    out=FIG_DIR/'figure_manifest.txt'
    with out.open('w') as f:
        for name,status,msg in manifest:
            f.write(f'{name}: {status}' + (f' — {msg}' if msg else '') + '\n')
    for name,status,msg in manifest:
        print(f'{name}: {status}' + (f' — {msg}' if msg else ''))

if __name__ == '__main__': main()
