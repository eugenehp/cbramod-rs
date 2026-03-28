#!/usr/bin/env python3
"""Benchmark CBraMod: Python (PyTorch) vs Rust (Burn)."""

import sys, types, time, json, os, subprocess, platform
import torch
import numpy as np

# Mock braindecode
class EEGModuleMixin:
    def __init__(self, n_outputs=None, n_chans=None, chs_info=None, n_times=None, input_window_seconds=None, sfreq=None, **kwargs):
        super().__init__()
        self.n_outputs = n_outputs; self.n_chans = n_chans; self.chs_info = chs_info; self.n_times = n_times; self.sfreq = sfreq

bmmb = types.ModuleType('braindecode.models.base')
bmmb.EEGModuleMixin = EEGModuleMixin
sys.modules['braindecode'] = types.ModuleType('braindecode')
sys.modules['braindecode.models'] = types.ModuleType('braindecode.models')
sys.modules['braindecode.models.base'] = bmmb
bfunc = types.ModuleType('braindecode.functional')
bfunc._get_gaussian_kernel1d = lambda *a, **kw: None
sys.modules['braindecode.functional'] = bfunc
bmods = types.ModuleType('braindecode.modules')
sys.modules['braindecode.modules'] = bmods

import importlib.util
att_spec = importlib.util.spec_from_file_location('attention', '/Users/Shared/braindecode/braindecode/modules/attention.py')
att_mod = importlib.util.module_from_spec(att_spec)
att_spec.loader.exec_module(att_mod)
bmods.CrissCrossTransformerEncoderLayer = att_mod.CrissCrossTransformerEncoderLayer

spec = importlib.util.spec_from_file_location('cbramod', '/Users/Shared/braindecode/braindecode/models/cbramod.py')
cbramod_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cbramod_mod)
CBraMod = cbramod_mod.CBraMod

WARMUP, REPEATS = 5, 30
CONFIGS = [
    (4,   1000, "4ch×1000t"),
    (8,   1000, "8ch×1000t"),
    (16,  1000, "16ch×1000t"),
    (22,  1000, "22ch×1000t"),
    (32,  1000, "32ch×1000t"),
    (64,  1000, "64ch×1000t"),
    (22,  2000, "22ch×2000t"),
    (22,  4000, "22ch×4000t"),
]
RUST_BACKENDS = [
    ("ndarray",    "target/release/examples/benchmark_ndarray"),
    ("accelerate", "target/release/examples/benchmark_accelerate"),
    ("metal",      "target/release/examples/benchmark_metal"),
]

def bench_python(n_chans, n_times):
    torch.manual_seed(42)
    model = CBraMod(n_outputs=4, n_chans=n_chans, n_times=n_times, sfreq=200,
                    patch_size=200, dim_feedforward=800, n_layer=2, nhead=8, emb_dim=200, drop_prob=0.0)
    model.eval()
    x = torch.randn(1, n_chans, n_times)
    with torch.no_grad():
        for _ in range(WARMUP): _ = model(x)
    times = []
    with torch.no_grad():
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            _ = model(x)
            times.append((time.perf_counter() - t0) * 1000)
    return times

def bench_rust(binary, n_chans, n_times):
    if not os.path.exists(binary): return None
    try:
        r = subprocess.run([binary, str(n_chans), str(n_times), str(WARMUP), str(REPEATS)],
                          capture_output=True, text=True, timeout=120)
        if r.returncode != 0: return None
        return json.loads(r.stdout)["times_ms"]
    except: return None

def main():
    os.makedirs("figures", exist_ok=True)
    results = {"meta": {"platform": platform.platform(), "machine": platform.machine(),
                        "torch_version": torch.__version__, "warmup": WARMUP, "repeats": REPEATS}, "benchmarks": []}

    print(f"Platform: {platform.platform()}")
    print(f"PyTorch: {torch.__version__}, depth=2, warmup={WARMUP}, repeats={REPEATS}\n")

    for n_chans, n_times, label in CONFIGS:
        print(f"── {label} ──")
        py_times = bench_python(n_chans, n_times)
        py_mean, py_std = np.mean(py_times), np.std(py_times)
        print(f"  Python (PyTorch):     {py_mean:7.2f} ± {py_std:.2f} ms")
        entry = {"label": label, "n_chans": n_chans, "n_times": n_times,
                 "python_times_ms": py_times, "python_mean_ms": float(py_mean), "python_std_ms": float(py_std)}
        for bk, binary in RUST_BACKENDS:
            rs = bench_rust(binary, n_chans, n_times)
            if rs:
                m, s = np.mean(rs), np.std(rs)
                sp = py_mean / m
                print(f"  Rust ({bk:12s}): {m:7.2f} ± {s:.2f} ms  ({sp:.2f}x)")
            else: m = s = sp = None; rs = []
            entry[f"rust_{bk}_times_ms"] = rs
            entry[f"rust_{bk}_mean_ms"] = float(m) if m else None
            entry[f"rust_{bk}_std_ms"] = float(s) if s else None
            entry[f"rust_{bk}_speedup"] = float(sp) if sp else None
        results["benchmarks"].append(entry)
        print()

    with open("figures/benchmark_results.json", "w") as f: json.dump(results, f, indent=2)
    generate_charts(results)

def generate_charts(results):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    B = results["benchmarks"]
    labels = [b["label"] for b in B]
    py_means = [b["python_mean_ms"] for b in B]
    py_stds = [b["python_std_ms"] for b in B]
    colors = {"python": "#4C72B0", "ndarray": "#DD8452", "accelerate": "#55A868", "metal": "#C44E52"}
    names = {"python": "Python (PyTorch)", "ndarray": "Rust (NdArray)", "accelerate": "Rust (Accelerate)", "metal": "Rust (Metal GPU)"}
    active = [bk for bk in ["ndarray","accelerate","metal"] if any(b.get(f"rust_{bk}_mean_ms") for b in B)]
    n_bars = 1 + len(active); width = 0.8 / n_bars; x = np.arange(len(labels))

    # Latency
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width*(n_bars-1)/2, py_means, width, yerr=py_stds, label=names["python"], color=colors["python"], capsize=2, alpha=0.85)
    for i, bk in enumerate(active):
        ms = [b.get(f"rust_{bk}_mean_ms") or 0 for b in B]
        ss = [b.get(f"rust_{bk}_std_ms") or 0 for b in B]
        ax.bar(x - width*(n_bars-1)/2 + width*(i+1), ms, width, yerr=ss, label=names[bk], color=colors[bk], capsize=2, alpha=0.85)
    ax.set_xlabel('Configuration'); ax.set_ylabel('Latency (ms)')
    ax.set_title('CBraMod Inference Latency', fontsize=14, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3); plt.tight_layout()
    plt.savefig('figures/inference_latency.png', dpi=150); plt.close()
    print("Saved figures/inference_latency.png")

    # Speedup
    fig, ax = plt.subplots(figsize=(14, 6))
    sp_w = 0.8 / max(len(active), 1)
    for i, bk in enumerate(active):
        sps = [b.get(f"rust_{bk}_speedup") or 0 for b in B]
        bars = ax.bar(x - sp_w*(len(active)-1)/2 + sp_w*i, sps, sp_w, color=colors[bk], alpha=0.85, label=names[bk])
        for bar, sp in zip(bars, sps):
            if sp > 0: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f'{sp:.2f}x', ha='center', va='bottom', fontsize=7, fontweight='bold')
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, label='Parity (1.0x)')
    ax.set_xlabel('Configuration'); ax.set_ylabel('Speedup (vs Python)')
    ax.set_title('Rust Speedup over Python (PyTorch)', fontsize=14, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3); plt.tight_layout()
    plt.savefig('figures/speedup.png', dpi=150); plt.close()
    print("Saved figures/speedup.png")

    # Channel scaling
    cb = [b for b in B if b["n_times"] == 1000]
    if len(cb) > 1:
        fig, ax = plt.subplots(figsize=(9, 5))
        ch = [b["n_chans"] for b in cb]
        ax.plot(ch, [b["python_mean_ms"] for b in cb], 'o-', color=colors["python"], label=names["python"], linewidth=2, markersize=7)
        for bk in active:
            la = [b.get(f"rust_{bk}_mean_ms") for b in cb]
            if any(v for v in la):
                ax.plot([c for c,v in zip(ch,la) if v], [v for v in la if v], 's-', color=colors[bk], label=names[bk], linewidth=2, markersize=7)
        ax.set_xlabel('Number of Channels'); ax.set_ylabel('Latency (ms)')
        ax.set_title('Latency vs Channel Count (T=1000)', fontsize=14, fontweight='bold')
        ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
        plt.savefig('figures/channel_scaling.png', dpi=150); plt.close()
        print("Saved figures/channel_scaling.png")

    # Time scaling
    tb = [b for b in B if b["n_chans"] == 22]
    if len(tb) > 1:
        fig, ax = plt.subplots(figsize=(9, 5))
        ts = [b["n_times"] for b in tb]
        ax.plot(ts, [b["python_mean_ms"] for b in tb], 'o-', color=colors["python"], label=names["python"], linewidth=2, markersize=7)
        for bk in active:
            la = [b.get(f"rust_{bk}_mean_ms") for b in tb]
            if any(v for v in la):
                ax.plot([t for t,v in zip(ts,la) if v], [v for v in la if v], 's-', color=colors[bk], label=names[bk], linewidth=2, markersize=7)
        ax.set_xlabel('Number of Time Samples'); ax.set_ylabel('Latency (ms)')
        ax.set_title('Latency vs Signal Length (C=22)', fontsize=14, fontweight='bold')
        ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
        plt.savefig('figures/time_scaling.png', dpi=150); plt.close()
        print("Saved figures/time_scaling.png")

if __name__ == "__main__":
    main()
