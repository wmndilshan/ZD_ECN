#!/usr/bin/env python3
"""
Procedural skin texture v2 for the skin_rendering app (cylindrical UV space).

Improvements over v1 (uniform multi-octave noise):
  - Body-region variation driven by the cylindrical UV parameterisation:
      head        (v > 0.85):          darken x0.85, slight cool shift
      hands       (u seam bands):      warm shift (+R, -B)
      inner torso (u ~ 0.5, v 0.3-0.7): lighten x1.05
  - Joint crease darkening: thin horizontal bands at v ~ 0.25 / 0.45 / 0.65
    (knees, hips, shoulders in cylindrical projection) darkened x0.80.
  - 3-octave value noise for natural variation (amplitude 12).

The texture is generated directly in UV space, so it is correct for any UV
parameterisation that covers [0,1]^2 (unlike the Meshcapade albedo, which is
authored for Meshcapade's own UV layout).

Also regenerates texture_skin_1 from the Meshcapade CC-BY albedo: the previous
centre crop grabbed the face/eyeball UV islands (the albedo is a *clothed*
avatar texture) which smeared eyes onto the legs under cylindrical UV. v2
mirror-tiles a small feature-free cheek-skin patch into a seamless texture, so
slot 1 shows real captured skin micro-detail without facial features.

Writes:
  skin_rendering/assets/texture_skin_0.png                       (repo asset copy)
  skin_rendering/android/app/src/main/assets/texture_skin_0.png  (app asset)
  skin_rendering/assets/texture_skin_1.png
  skin_rendering/android/app/src/main/assets/texture_skin_1.png
"""
from __future__ import annotations

import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_REPO = ROOT / "skin_rendering/assets/texture_skin_0.png"
OUT_APP  = ROOT / "skin_rendering/android/app/src/main/assets/texture_skin_0.png"
MESHCAPADE_ALBEDO = ROOT / "reference/smpl_texture_samples/Textured_Mesh_samples/f_01_alb.002.png"
OUT1_REPO = ROOT / "skin_rendering/assets/texture_skin_1.png"
OUT1_APP  = ROOT / "skin_rendering/android/app/src/main/assets/texture_skin_1.png"

SIZE = 512
SEED = 42
BASE = np.array([210.0, 165.0, 120.0])  # neutral warm skin


def value_noise(size: int, freq: int, rng: np.random.Generator) -> np.ndarray:
    """Smooth value noise: random grid bilinearly upsampled; tiles on the u axis."""
    grid = rng.uniform(-1.0, 1.0, (freq + 1, freq + 1)).astype(np.float32)
    grid[:, -1] = grid[:, 0]  # tile horizontally (u wraps at the cylindrical seam)
    coords = np.linspace(0, freq, size, endpoint=False)
    xi = coords.astype(int)
    xf = (coords - xi).astype(np.float32)
    # smoothstep interpolation weights
    wx = xf * xf * (3.0 - 2.0 * xf)
    v00 = grid[np.ix_(xi, xi)]
    v01 = grid[np.ix_(xi, xi + 1)]
    v10 = grid[np.ix_(xi + 1, xi)]
    v11 = grid[np.ix_(xi + 1, xi + 1)]
    wy = wx[:, None]
    wxr = wx[None, :]
    return (v00 * (1 - wy) * (1 - wxr) + v01 * (1 - wy) * wxr
            + v10 * wy * (1 - wxr) + v11 * wy * wxr)


def main() -> None:
    rng = np.random.default_rng(SEED)
    u = np.linspace(0, 1, SIZE, endpoint=False)[None, :]   # column = u
    # image row 0 is the TOP of the texture; cylindrical v=1 is the head,
    # and GL samples v directly, so image row r corresponds to v = 1 - r/SIZE
    v = np.linspace(1, 0, SIZE, endpoint=False)[:, None]   # row = v (top=head)

    rgb = np.broadcast_to(BASE, (SIZE, SIZE, 3)).astype(np.float32).copy()

    # ── 3-octave value noise, amplitude 12 ─────────────────────────────────
    noise = np.zeros((SIZE, SIZE), dtype=np.float32)
    for freq, amp in [(8, 12.0), (24, 6.0), (64, 3.0)]:
        noise += amp * value_noise(SIZE, freq, rng)
    rgb += noise[:, :, None] * np.array([1.0, 0.9, 0.75])

    # ── head region (v > 0.85): darken, slight cool shift ─────────────────
    head = np.clip((v - 0.85) / 0.05, 0.0, 1.0)  # smooth ramp 0.85→0.90
    head_mult = 1.0 - 0.15 * head                # ×0.85 at full strength
    rgb *= head_mult[:, :, None]
    rgb[:, :, 0] -= 4.0 * head                   # cool shift: less red
    rgb[:, :, 2] += 3.0 * head                   #             more blue

    # ── hand regions (u seam bands): warm shift +R8 −B5 ───────────────────
    # Arms hang at the body sides: in cylindrical projection the hands sit
    # near u ≈ 0.25 and u ≈ 0.75 in the lower-mid band (v 0.35–0.55).
    hand_u = (np.exp(-0.5 * ((u - 0.25) / 0.05) ** 2)
              + np.exp(-0.5 * ((u - 0.75) / 0.05) ** 2))
    hand_v = np.exp(-0.5 * ((v - 0.45) / 0.08) ** 2)
    hands = np.clip(hand_u * hand_v, 0.0, 1.0)
    rgb[:, :, 0] += 8.0 * hands
    rgb[:, :, 2] -= 5.0 * hands

    # ── inner torso (u near 0.5, v 0.3–0.7): lighten ×1.05 ────────────────
    torso_u = np.exp(-0.5 * ((u - 0.5) / 0.12) ** 2)
    torso_v = np.clip((v - 0.3) / 0.05, 0.0, 1.0) * np.clip((0.7 - v) / 0.05, 0.0, 1.0)
    torso = np.clip(torso_u * torso_v, 0.0, 1.0)
    rgb *= (1.0 + 0.05 * torso)[:, :, None]

    # ── joint crease darkening: thin bands at knees/hips/shoulders ────────
    crease = np.zeros_like(v)
    for vc in (0.25, 0.45, 0.65):
        crease += np.exp(-0.5 * ((v - vc) / 0.02) ** 2)
    crease = np.clip(crease, 0.0, 1.0)
    # modulate by noise so the crease is broken up, not a hard drawn line
    crease2d = crease * (0.6 + 0.4 * np.clip(noise / 12.0 + 0.5, 0.0, 1.0))
    rgb *= (1.0 - 0.12 * crease2d)[:, :, None]   # subtle darkening at band centre

    rgb = np.clip(rgb, 60, 255).astype(np.uint8)

    # ── save RGBA PNG ──────────────────────────────────────────────────────
    rgba = np.dstack([rgb, np.full((SIZE, SIZE), 255, dtype=np.uint8)])
    try:
        from PIL import Image
        img = Image.fromarray(rgba, "RGBA")
        for out in (OUT_REPO, OUT_APP):
            out.parent.mkdir(parents=True, exist_ok=True)
            img.save(out)
            print(f"wrote {out.relative_to(ROOT)}  ({SIZE}x{SIZE} RGBA)")
    except ImportError:
        import zlib, struct
        raw = b"".join(b"\x00" + row.tobytes() for row in rgba)
        def chunk(name, data):
            crc = zlib.crc32(name + data) & 0xFFFFFFFF
            return struct.pack(">I", len(data)) + name + data + struct.pack(">I", crc)
        png = (b"\x89PNG\r\n\x1a\n"
               + chunk(b"IHDR", struct.pack(">IIBBBBB", SIZE, SIZE, 8, 6, 0, 0, 0))
               + chunk(b"IDAT", zlib.compress(raw, 9))
               + chunk(b"IEND", b""))
        for out in (OUT_REPO, OUT_APP):
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(png)
            print(f"wrote {out.relative_to(ROOT)}  ({SIZE}x{SIZE} RGBA, stdlib)")

    print(f"mean tone: {rgb.reshape(-1,3).mean(axis=0).round(1)}")

    generate_meshcapade_tile()


def generate_meshcapade_tile() -> None:
    """texture_skin_1: seamless tile from a feature-free Meshcapade skin patch."""
    if not MESHCAPADE_ALBEDO.exists():
        print(f"SKIP texture_skin_1: {MESHCAPADE_ALBEDO} not found")
        return
    try:
        from PIL import Image
    except ImportError:
        print("SKIP texture_skin_1: Pillow required for Meshcapade tile")
        return
    img = Image.open(MESHCAPADE_ALBEDO)
    # Lower-left cheek, right of the nostril marks and above the lips —
    # the largest bare-skin region in this clothed-avatar albedo.
    strip = np.array(img.crop((1330, 995, 1500, 1075)), dtype=np.uint8)  # 80×170
    v = np.concatenate([strip, strip[::-1]], axis=0)     # mirror: seamless vertically
    v = np.tile(v, (4, 1, 1))
    h = np.concatenate([v, v[:, ::-1]], axis=1)          # mirror: seamless horizontally
    h = np.tile(h, (1, 2, 1))
    tile = h[:SIZE, :SIZE]
    rgba = np.dstack([tile, np.full((SIZE, SIZE), 255, dtype=np.uint8)])
    out_img = Image.fromarray(rgba, "RGBA")
    for out in (OUT1_REPO, OUT1_APP):
        out.parent.mkdir(parents=True, exist_ok=True)
        out_img.save(out)
        print(f"wrote {out.relative_to(ROOT)}  ({SIZE}x{SIZE} RGBA, Meshcapade CC-BY tile)")


if __name__ == "__main__":
    main()
