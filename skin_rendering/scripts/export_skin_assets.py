#!/usr/bin/env python3
"""
Generate all binary and image assets required by the skin_rendering Android app.

Outputs (relative to repo root):
  skin_rendering/android/app/src/main/assets/
    smpl_uv_cylindrical_f32.bin   (6890 × 2 float32, cylindrical UV)
    texture_skin_0.png             (512×512 procedural neutral skin)
    texture_skin_1.png             (512×512 Meshcapade albedo crop, or procedural)
    zdecn_weights_fp16.bin         (symlink/copy from android_smpl_app assets)
    animation_poses_aa_f32.bin     (copy)
    animation_poses_rot6d_f32.bin  (copy)
    smpl/                          (copy directory from android_smpl_app)
    smpl_shader_data/              (copy directory from android_smpl_app)
    shaders/                       (copied below by build script)
"""
from __future__ import annotations

import argparse
import math
import pickle
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("WARNING: Pillow not available — procedural texture generated with numpy only (PNG via stdlib)")

ROOT = Path(__file__).resolve().parent.parent.parent
SMPL_APP_ASSETS = ROOT / "android_smpl_app/app/src/main/assets"
SKIN_ASSETS     = ROOT / "skin_rendering/android/app/src/main/assets"


# ── helpers ──────────────────────────────────────────────────────────────────

def save_png_numpy(path: Path, rgb: np.ndarray) -> None:
    """Minimal PNG writer using Python stdlib (no Pillow fallback)."""
    import zlib, struct as _s
    h, w, c = rgb.shape
    assert c == 3
    raw = b""
    for row in rgb.astype(np.uint8):
        raw += b"\x00" + row.tobytes()
    compressed = zlib.compress(raw, 9)
    def chunk(name, data):
        crc = zlib.crc32(name + data) & 0xFFFFFFFF
        return _s.pack(">I", len(data)) + name + data + _s.pack(">I", crc)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", _s.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)))
        f.write(chunk(b"IDAT", compressed))
        f.write(chunk(b"IEND", b""))


def make_procedural_skin(size: int = 512, seed: int = 42) -> np.ndarray:
    """
    Generate a procedural skin texture (RGB, uint8).
    Base: warm neutral skin tone (210, 165, 120).
    Adds multi-frequency sinusoidal variation to mimic skin pore texture.
    Slightly darker in the centre (simulates subsurface depth gradient).
    """
    rng = np.random.default_rng(seed)
    u = np.linspace(0, 1, size, endpoint=False)
    v = np.linspace(0, 1, size, endpoint=False)
    uu, vv = np.meshgrid(u, v)

    # base skin tone (warm neutral)
    base = np.array([210.0, 165.0, 120.0])

    # multi-frequency noise (4 octaves of sinusoidal approximation)
    noise = np.zeros((size, size), dtype=np.float32)
    for freq, amp in [(8, 6), (16, 3), (32, 1.5), (64, 0.8)]:
        phase_u = rng.uniform(0, 2 * math.pi)
        phase_v = rng.uniform(0, 2 * math.pi)
        noise += amp * np.sin(freq * 2 * math.pi * uu + phase_u) * np.cos(freq * 2 * math.pi * vv + phase_v)

    # pore-like detail (sparse random dots)
    pores = rng.standard_normal((size, size)).astype(np.float32) * 3.0

    variation = noise + pores * 0.4

    rgb = np.zeros((size, size, 3), dtype=np.float32)
    for c, (b, scale) in enumerate(zip(base, [1.0, 1.0, 0.85])):
        rgb[:, :, c] = np.clip(b + variation * scale, 80, 255)

    return rgb.astype(np.uint8)


def make_texture_from_albedo(albedo_path: Path, size: int = 512) -> np.ndarray:
    """
    Crop and resize the Meshcapade albedo texture.
    The albedo is a 4096×4096 UV-space texture; we crop the most skin-dense
    region (centre 40% = rows 1600..2457, cols 1600..2457) which corresponds
    to the torso UV island, then resize to 512×512.
    """
    if not HAS_PIL:
        return None
    img = Image.open(albedo_path).convert("RGB")
    w, h = img.size  # (4096, 4096)
    cx, cy = w // 2, h // 2
    half = int(min(w, h) * 0.2)
    crop = img.crop((cx - half, cy - half, cx + half, cy + half))
    return np.array(crop.resize((size, size), Image.LANCZOS), dtype=np.uint8)


# ── UV generation ─────────────────────────────────────────────────────────────

def compute_cylindrical_uv(v_template: np.ndarray) -> np.ndarray:
    """
    Compute cylindrical UV coordinates for SMPL's 6890 vertices.

    u = atan2(x, z) / (2π) + 0.5   — horizontal, wraps at 0/1
    v = (y - y_min) / (y_max - y_min)  — vertical coverage

    UV range: both u and v in [0, 1].
    No seam discontinuity because u is computed per-vertex and GL_REPEAT handles
    the 0/1 boundary seamlessly for a closed cylindrical surface.

    Returns float32 array of shape (6890, 2).
    """
    x = v_template[:, 0].astype(np.float64)
    y = v_template[:, 1].astype(np.float64)
    z = v_template[:, 2].astype(np.float64)

    u = np.arctan2(x, z) / (2.0 * math.pi) + 0.5   # [0, 1]
    y_min, y_max = y.min(), y.max()
    v = (y - y_min) / (y_max - y_min)               # [0, 1]

    uv = np.stack([u, v], axis=1).astype(np.float32)
    return uv


# ── copy helpers ──────────────────────────────────────────────────────────────

def copy_asset_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"  copied  {dst.relative_to(ROOT)}")


def copy_asset_dir(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    n = sum(1 for _ in dst.rglob("*") if _.is_file())
    print(f"  copied  {dst.relative_to(ROOT)}/ ({n} files)")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smpl-pkl",  type=Path,
                    default=ROOT / "models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl")
    ap.add_argument("--texture-size", type=int, default=512)
    args = ap.parse_args()

    SKIN_ASSETS.mkdir(parents=True, exist_ok=True)

    # ── 1. UV coordinates ─────────────────────────────────────────────────────
    print("Loading SMPL template …")
    with open(args.smpl_pkl, "rb") as f:
        smpl = pickle.load(f, encoding="latin1")
    v_template = np.array(smpl["v_template"], dtype=np.float32)   # (6890, 3)
    print(f"  v_template: {v_template.shape}  y=[{v_template[:,1].min():.3f}, {v_template[:,1].max():.3f}]")

    uv = compute_cylindrical_uv(v_template)
    uv_path = SKIN_ASSETS / "smpl_uv_cylindrical_f32.bin"
    uv_path.write_bytes(uv.tobytes())
    print(f"  UV saved: {uv_path.relative_to(ROOT)}  shape={uv.shape}")

    # ── 2. Textures ───────────────────────────────────────────────────────────
    sz = args.texture_size
    albedo_f = ROOT / "reference/smpl_texture_samples/Textured_Mesh_samples/f_01_alb.002.png"
    albedo_m = ROOT / "reference/smpl_texture_samples/Textured_Mesh_samples/m_01_alb.002.png"

    # Texture 0: procedural neutral skin (always available)
    print("Generating texture_skin_0 (procedural) …")
    tex0 = make_procedural_skin(sz)

    # Texture 1: Meshcapade albedo crop, or another procedural variation
    print("Generating texture_skin_1 …")
    tex1 = None
    if HAS_PIL and albedo_f.exists():
        tex1 = make_texture_from_albedo(albedo_f, sz)
        print(f"  source: {albedo_f.name} (centre crop {sz}×{sz})")
    elif HAS_PIL and albedo_m.exists():
        tex1 = make_texture_from_albedo(albedo_m, sz)
        print(f"  source: {albedo_m.name} (centre crop {sz}×{sz})")
    if tex1 is None:
        tex1 = make_procedural_skin(sz, seed=7)
        print("  source: procedural (Meshcapade not found or Pillow missing)")

    for i, tex in enumerate([tex0, tex1]):
        path = SKIN_ASSETS / f"texture_skin_{i}.png"
        if HAS_PIL:
            Image.fromarray(tex).save(path)
        else:
            save_png_numpy(path, tex)
        print(f"  saved texture_{i}: {path.relative_to(ROOT)}")

    # ── 3. Copy ZD-ECN weights ────────────────────────────────────────────────
    print("Copying weights and animation data …")
    for fname in ["zdecn_weights_fp16.bin", "zdecn_weights_fp32.bin",
                  "animation_poses_aa_f32.bin", "animation_poses_rot6d_f32.bin"]:
        src = SMPL_APP_ASSETS / fname
        if src.exists():
            copy_asset_file(src, SKIN_ASSETS / fname)
        else:
            print(f"  WARNING: {src} not found — skipping")

    # ── 4. Copy SMPL model binary data ───────────────────────────────────────
    for subdir in ["smpl", "smpl_shader_data"]:
        src = SMPL_APP_ASSETS / subdir
        if src.exists():
            copy_asset_dir(src, SKIN_ASSETS / subdir)
        else:
            print(f"  WARNING: {src}/ not found")

    # ── 5. Copy ONNX model ───────────────────────────────────────────────────
    for fname in ["zdecn_fp16.onnx"]:
        src = SMPL_APP_ASSETS / fname
        if src.exists():
            copy_asset_file(src, SKIN_ASSETS / fname)

    print()
    print("═" * 55)
    print("  skin_rendering assets ready")
    print(f"  UV:      {uv.shape} cylindrical, range [{uv[:,0].min():.3f},{uv[:,0].max():.3f}]")
    print(f"  tex0:    {tex0.shape} procedural skin")
    print(f"  tex1:    {tex1.shape} {'Meshcapade crop' if (HAS_PIL and albedo_f.exists()) else 'procedural'}")
    print("═" * 55)


if __name__ == "__main__":
    main()
