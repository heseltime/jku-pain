# bump/views.py
import io, os, hashlib, re
from math import pi, asin, degrees
import numpy as np
import random
from PIL import Image
from django.http import HttpResponse, JsonResponse
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from django.conf import settings
import colorsys
from typing import Sequence, Tuple
from django.urls import reverse

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

PLANETARY_FALLBACK_ENABLED = True
OPENAI_TEXT_MODEL = "gpt-4.1-mini"  # or your preferred model
# Ensure OPENAI_API_KEY is loaded (via dotenv or environment)

EARTH_SRC_PATH = os.getenv(
    "EARTH_TEXTURE_SRC",
    str((settings.BASE_DIR / "assets" / "8k_earth_daymap_greyscale.jpg").resolve())
)

import io
import os
import hashlib
from PIL import Image, ImageOps, ImageFilter
from django.http import HttpResponse
from django.conf import settings
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
import numpy as np
import requests

# ---------- default server-side image ----------
EARTH_BUMPMAP_MASK_PATH = os.getenv(
    "EARTH_BUMPMAP_MASK_SRC",
    #str((settings.BASE_DIR / "assets" / "8k_earth_daymap_greyscale.jpg").resolve())
    str((settings.BASE_DIR / "assets" / "standin-economic-data.jpeg").resolve())
)

# ---------- helpers ----------
def _parse_bool(v, default=False):
    return str(v).lower() in ("1","true","t","yes","y","on") if v is not None else default

def _read_file_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()

def _load_source_image_bytes(request):
    """
    Priority:
      1) multipart 'mask' (file upload)
      2) query param 'mask_url'
      3) query param 'mask_path' (server filesystem)
      4) EARTH_BUMPMAP_MASK_PATH (server default)
    Returns (bytes, etag_bytes_hint)
    """
    # 1) file upload
    if 'mask' in request.FILES:
        b = request.FILES['mask'].read()
        return b, hashlib.md5(b).digest()

    # 2) URL
    mask_url = request.GET.get('mask_url')
    if mask_url:
        r = requests.get(mask_url, timeout=10)
        r.raise_for_status()
        b = r.content
        return b, hashlib.md5(b).digest()

    # 3) server path
    mask_path = request.GET.get('mask_path')
    if mask_path and os.path.isfile(mask_path):
        b = _read_file_bytes(mask_path)
        return b, hashlib.md5(b).digest()

    # 4) default
    if os.path.isfile(EARTH_BUMPMAP_MASK_PATH):
        b = _read_file_bytes(EARTH_BUMPMAP_MASK_PATH)
        return b, hashlib.md5(b).digest()

    # fallback: tiny white image to avoid 500s
    im = Image.new("L", (512, 256), 255)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    b = buf.getvalue()
    return b, hashlib.md5(b).digest()

def _auto_levels_u8(gray_u8: np.ndarray) -> np.ndarray:
    """Stretch histogram to full [0,255] via per-image min/max."""
    lo = int(gray_u8.min())
    hi = int(gray_u8.max())
    if hi <= lo:
        return gray_u8
    # scale to 0..255
    out = (gray_u8.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(out, 0, 255).astype(np.uint8)

# ---------- endpoint ----------
@api_view(["GET", "POST", "HEAD", "OPTIONS"])
@parser_classes([MultiPartParser, FormParser])
def bumpmap(request):
    """
    Fast bumpmap: grayscale + gaussian blur.

    Params:
      - w, h: output size. If omitted -> keep source size.
      - blur: Gaussian blur radius in pixels (float, default 6.0)
      - gamma: apply gamma correction (default 1.0; >1.0 darkens midtones)
      - invert: 0/1 flip grayscale
      - normalize: 0/1 scale to [0,1] before output (kept implicit in 8-bit)
      - auto_levels: 0/1 stretch to full [0,255] after blur (default 1)
      - hard: 0/1 apply threshold at the end
      - threshold: 0..1 threshold if hard=1 (default 0.35)
      - fmt: png|jpg|jpeg (default png)
      - seed: ignored (kept for parity)
      - mask/mask_url/mask_path: input source overrides
    """
    fmt = (request.GET.get("fmt", request.POST.get("fmt", "png")) or "png").lower()
    blur = float(request.GET.get("blur", request.POST.get("blur", 6.0)))
    gamma = float(request.GET.get("gamma", request.POST.get("gamma", 1.0)))
    invert = _parse_bool(request.GET.get("invert", request.POST.get("invert")), False)
    auto_levels = _parse_bool(request.GET.get("auto_levels", request.POST.get("auto_levels", 1)), True)
    hard = _parse_bool(request.GET.get("hard", request.POST.get("hard")), False)
    threshold = float(request.GET.get("threshold", request.POST.get("threshold", 0.35)))

    # Load source bytes and a stable hash for ETag
    src_bytes, src_hash = _load_source_image_bytes(request)

    # Open image
    with Image.open(io.BytesIO(src_bytes)) as im_src:
        im_src = im_src.convert("RGB")  # robust
        # If w/h provided -> resize; else keep source size
        w_param = request.GET.get("w", request.POST.get("w"))
        h_param = request.GET.get("h", request.POST.get("h"))

        if w_param and h_param:
            w = int(w_param); h = int(h_param)
            im_src = im_src.resize((w, h), Image.LANCZOS if hasattr(Image, "LANCZOS") else Image.BICUBIC)
        else:
            w, h = im_src.size

        # Convert to grayscale
        im_gray = ImageOps.grayscale(im_src)  # 'L'

        # Optional gamma (convert to float for precision)
        if gamma and gamma != 1.0:
            arr = np.asarray(im_gray, dtype=np.uint8)
            arrf = (arr.astype(np.float32) / 255.0) ** np.float32(gamma)
            im_gray = Image.fromarray(np.clip(arrf * 255.0, 0, 255).astype(np.uint8), mode="L")

        # Invert if requested
        if invert:
            im_gray = ImageOps.invert(im_gray)

        # Gaussian blur (fast)
        blur = max(0.0, float(blur))
        if blur > 0.0:
            im_gray = im_gray.filter(ImageFilter.GaussianBlur(radius=blur))

        # Auto-levels to use full dynamic range (improves bump contrast)
        if auto_levels:
            arr = np.asarray(im_gray, dtype=np.uint8)
            im_gray = Image.fromarray(_auto_levels_u8(arr), mode="L")

        # Optional hard threshold for stylized masks
        if hard:
            thr = int(np.clip(threshold, 0.0, 1.0) * 255.0)
            im_gray = im_gray.point(lambda v, t=thr: 255 if v >= t else 0, mode='L')

        # Encode
        buf = io.BytesIO()
        if fmt in ("jpg", "jpeg"):
            im_gray.save(buf, format="JPEG", quality=95, subsampling=0)
            ctype = "image/jpeg"
        else:
            im_gray.save(buf, format="PNG", optimize=True)
            ctype = "image/png"
        body = buf.getvalue()

    # ETag with params
    etag_src = b"|".join([
        src_hash,
        f"{w}x{h}".encode(),
        f"blur={blur}".encode(),
        f"gamma={gamma}".encode(),
        f"invert={int(invert)}".encode(),
        f"autolvl={int(auto_levels)}".encode(),
        f"hard={int(hard)}".encode(),
        f"thr={threshold}".encode(),
        f"fmt={fmt}".encode(),
    ])
    etag = hashlib.md5(etag_src).hexdigest()

    resp = HttpResponse(b"" if request.method == "HEAD" else body, content_type=ctype)
    resp["Content-Length"] = str(len(body))
    resp["ETag"] = etag
    resp["Cache-Control"] = "public, max-age=3600"
    return resp

# --- Cloud map endpoint -----------------------------------------------------
import io, hashlib, time
import numpy as np
from PIL import Image
from rest_framework.decorators import api_view
from django.http import HttpResponse

# Perlin helpers (row-wise, memory-safe)
def _perm_table(seed: int):
    rng = np.random.default_rng(seed)
    p = np.arange(256, dtype=np.int32)
    rng.shuffle(p)
    p = np.concatenate([p, p])
    return p

def _fade(t):  # 6t^5 - 15t^4 + 10t^3
    return t * t * t * (t * (t * 6 - 15) + 10)

def _grad(h, x, y):
    # simple gradient hash (±x ±y)
    u = np.where((h & 1) == 0, x, -x)
    v = np.where((h & 2) == 0, y, -y)
    return u + v

def _perlin2d(x, y, p):
    xi = (np.floor(x).astype(np.int32) & 255)
    yi = (np.floor(y).astype(np.int32) & 255)
    xf = x - np.floor(x)
    yf = y - np.floor(y)

    u = _fade(xf)
    v = _fade(yf)

    aa = p[p[xi] + yi]
    ab = p[p[xi] + yi + 1]
    ba = p[p[xi + 1] + yi]
    bb = p[p[xi + 1] + yi + 1]

    n00 = _grad(aa, xf, yf)
    n10 = _grad(ba, xf - 1, yf)
    n01 = _grad(ab, xf, yf - 1)
    n11 = _grad(bb, xf - 1, yf - 1)

    x1 = n00 + u * (n10 - n00)
    x2 = n01 + u * (n11 - n01)
    return x1 + v * (x2 - x1)  # ~[-1,1]

def _fbm_row(xs, y, p, octaves=6, lac=2.0, gain=0.5, freq=1.0):
    total = np.zeros_like(xs, dtype=np.float32)
    amp = 1.0
    norm = 0.0
    f = freq
    for _ in range(octaves):
        total += amp * _perlin2d(xs * f, np.full_like(xs, y * f), p)
        norm += amp
        amp *= gain
        f *= lac
    return (total / max(norm, 1e-6)) * 0.5 + 0.5  # 0..1

def _apply_contrast(v, contrast=1.4, gamma=1.0):
    # contrast around 0.5, then gamma curve
    v = (v - 0.5) * contrast + 0.5
    v = np.clip(v, 0.0, 1.0)
    if gamma != 1.0:
        v = np.power(v, gamma, dtype=np.float32)
    return v

def _heat_color(anom):
    """
    Map anomaly (°C) to warm tint.
    0 -> white, 1.5 -> peach, 2.0 -> orange, 3.0 -> red, 4+ -> deep red.
    Returns np.float32 RGB in 0..1
    """
    stops = np.array([0.0, 1.5, 2.0, 3.0, 4.0], dtype=np.float32)
    cols = np.array([
        [1.00, 1.00, 1.00],
        [1.00, 0.96, 0.86],
        [1.00, 0.80, 0.45],
        [1.00, 0.40, 0.25],
        [0.90, 0.00, 0.00],
    ], dtype=np.float32)
    a = np.float32(np.clip(anom, 0.0, 4.0))
    i = int(np.searchsorted(stops, a) - 1)
    i = max(0, min(i, len(stops) - 2))
    t = (a - stops[i]) / (stops[i + 1] - stops[i] + 1e-6)
    return cols[i] * (1 - t) + cols[i + 1] * t  # RGB 0..1


@api_view(["GET", "HEAD", "OPTIONS"])
def cloudmap(request):
    """
    Cloud texture tinted by temperature anomaly.

    Query params:
      w,h            Image size (default 8192x4096, 2:1 equirectangular)
      seed           RNG seed (int). Omit for deterministic default.
      octaves        fBm octaves (default 6)
      lacunarity     frequency multiplier per octave (default 2.0)
      gain           amplitude falloff per octave (default 0.5)
      freq           base frequency (default 1.0) — higher => finer detail
      contrast       contrast around 0.5 (default 1.4)
      gamma          gamma curve (default 1.0)
      cover          0..1 soft threshold bias (default 0.0; positive => more clouds)
      anom           temperature anomaly in °C (default 1.2)
      alpha          0|1 return RGBA with alpha=v (default 0)
      fmt            png|jpg|jpeg (default png)
    """
    w = int(request.GET.get("w", 8192))
    h = int(request.GET.get("h", 4096))
    seed = request.GET.get("seed")
    seed = int(seed) if seed is not None else 1337
    octaves = int(request.GET.get("octaves", 6))
    lac = float(request.GET.get("lacunarity", 2.0))
    gain = float(request.GET.get("gain", 0.5))
    freq = float(request.GET.get("freq", 1.0))
    contrast = float(request.GET.get("contrast", 1.4))
    gamma = float(request.GET.get("gamma", 1.0))
    cover = float(request.GET.get("cover", 0.0))  # bias
    anom = float(request.GET.get("anom", 1.2))
    alpha_on = str(request.GET.get("alpha", "0")).lower() in ("1","true","t","yes","y","on")
    fmt = (request.GET.get("fmt", "png") or "png").lower()

    p = _perm_table(seed)
    xs = np.linspace(0, 1, w, dtype=np.float32)
    out = np.empty((h, w, 4 if alpha_on else 3), dtype=np.uint8)

    tint = _heat_color(anom).astype(np.float32)  # 3

    for i in range(h):
        y = np.float32(i / max(h - 1, 1))
        v = _fbm_row(xs, y, p, octaves=octaves, lac=lac, gain=gain, freq=freq)
        v = np.clip(v + cover, 0.0, 1.0)          # coverage bias
        v = _apply_contrast(v, contrast=contrast, gamma=gamma)

        # colorize: grayscale clouds * warm tint
        row_rgb = (v[:, None] * tint[None, :] * 255.0).astype(np.uint8)
        if alpha_on:
            a = (v * 255.0).astype(np.uint8)      # alpha = brightness
            out[i] = np.concatenate([row_rgb, a[:, None]], axis=1)
        else:
            out[i] = row_rgb

    mode = "RGBA" if alpha_on else "RGB"
    img = Image.fromarray(out, mode=mode)

    buf = io.BytesIO()
    if fmt in ("jpg", "jpeg"):
        img.save(buf, format="JPEG", quality=95, subsampling=0)
        ctype = "image/jpeg"
    else:
        img.save(buf, format="PNG", optimize=True)
        ctype = "image/png"
    body = buf.getvalue()

    etag = hashlib.md5(
        f"{w}x{h}:{seed}:{octaves}:{lac}:{gain}:{freq}:{contrast}:{gamma}:{cover}:{anom}:{alpha_on}:{fmt}".encode()
    ).hexdigest()

    resp = HttpResponse(b"" if request.method == "HEAD" else body, content_type=ctype)
    resp["Content-Length"] = str(len(body))
    resp["ETag"] = etag
    resp["Cache-Control"] = "public, max-age=3600"
    return resp


def _cmy_rainbow_lut(n=256, pastel=0.35, low_k=0.06, high_k=0.04):
    import numpy as np
    # --- sRGB <-> linear helpers ---
    def srgb_to_linear(c):
        a = 0.055
        return np.where(c <= 0.04045, c/12.92, ((c + a)/(1 + a))**2.4)
    def linear_to_srgb(c):
        a = 0.055
        return np.where(c <= 0.0031308, 12.92*c, (1 + a)*(c**(1/2.4)) - a)
    def mix(a, b, t): return a*(1.0 - t) + b*t

    # base CMY in sRGB
    cyan, magenta, yellow = [np.array(v, np.float32)
        for v in ((0,1,1), (1,0,1), (1,1,0))]
    black = np.zeros(3, np.float32)
    white = np.ones(3,  np.float32)

    # pastelize anchors by mixing with white
    cyan_p    = mix(cyan,    white, pastel)
    magenta_p = mix(magenta, white, pastel)
    yellow_p  = mix(yellow,  white, pastel)

    # calmer extremes
    dark_teal  = mix(black, cyan_p, 0.65)     # deep low end
    near_white = mix(white, yellow_p, 0.85)   # warm high end

    # control points (0..1): 5 stops → 4 segments
    pos = np.array([0.0, low_k + 0.18, 0.50, 1.0 - high_k - 0.18, 1.0], np.float32)
    stops_lin = [srgb_to_linear(x) for x in (dark_teal, cyan_p, magenta_p, yellow_p, near_white)]

    xs = np.linspace(0, 1, n, dtype=np.float32)
    lut = np.zeros((n, 3), dtype=np.float32)
    for i in range(4):
        a, b = pos[i], pos[i+1]
        m = (xs >= a) & (xs <= b)
        if not np.any(m): continue
        t = (xs[m] - a) / (b - a + 1e-8)
        lut[m] = stops_lin[i]*(1-t)[:,None] + stops_lin[i+1]*t[:,None]

    lut = np.clip(linear_to_srgb(lut), 0, 1)
    return (lut*255 + 0.5).astype(np.uint8)

def _apply_lut_chunk(gray_chunk: np.ndarray, lut: np.ndarray, gamma: float = 1.0, invert: bool = False):
    if gamma != 1.0:
        idx = ((gray_chunk.astype(np.float32) / 255.0) ** np.float32(gamma) * 255.0).astype(np.uint8)
    else:
        idx = gray_chunk
    if invert:
        idx = 255 - idx
    return lut[idx]  # (H,W,3)

@api_view(["GET", "HEAD", "OPTIONS"])
def earthtexture(request):
    """
    Subtle multi-hue colorization of the grayscale Earth texture.

    Query params:
      w,h         Output size (default 8192x4096)
      seed        RNG seed (default 1234)
      nstops      3..12 color stops (default 6)
      hue_span    0..1 hue range (default 0.18) — smaller = more subtle
      vibrance    saturation multiplier (default 0.7)
      bright      value multiplier (default 1.0)
      gamma       grayscale gamma before LUT (default 1.0)
      invert      0|1 invert grayscale (default 0)
      strength    0..1 blend of color vs. grayscale (default 0.35)
      alpha       0|1 add alpha=original grayscale (default 0)
      fmt         png|jpg|jpeg (default png)
    """
    w = int(request.GET.get("w", 8192))
    h = int(request.GET.get("h", 4096))
    seed = int(request.GET.get("seed", 1234))
    nstops = int(request.GET.get("nstops", 6))
    hue_span = float(request.GET.get("hue_span", 0.18))
    vibrance = float(request.GET.get("vibrance", 0.7))
    bright = float(request.GET.get("bright", 1.0))
    gamma = float(request.GET.get("gamma", 1.0))
    invert = str(request.GET.get("invert", "0")).lower() in ("1","true","t","yes","y","on")
    strength = float(request.GET.get("strength", 0.35))
    strength = float(np.clip(strength, 0.0, 1.0))
    alpha_on = str(request.GET.get("alpha", "0")).lower() in ("1","true","t","yes","y","on")
    fmt = (request.GET.get("fmt", "png") or "png").lower()

    # Load & resize grayscale
    img = Image.open(EARTH_SRC_PATH).convert("L")
    if img.size != (w, h):
        img = img.resize((w, h), Image.LANCZOS)

    lut = _cmy_rainbow_lut(n=256, pastel=float(request.GET.get("pastel", 0.35)))

    chunk = 256
    out = np.empty((h, w, 4 if alpha_on else 3), dtype=np.uint8)
    for y0 in range(0, h, chunk):
        y1 = min(h, y0 + chunk)
        g = np.asarray(img.crop((0, y0, w, y1)), dtype=np.uint8)  # (rows, w)
        rgb = _apply_lut_chunk(g, lut, gamma=gamma, invert=invert)  # (rows, w, 3)

        # SUBTLE: blend back toward grayscale by 'strength'
        if strength < 1.0:
            gray3 = np.repeat(g[..., None], 3, axis=2)
            rgb = ((1.0 - strength) * gray3 + strength * rgb).astype(np.uint8)

        if alpha_on:
            out[y0:y1] = np.concatenate([rgb, g[..., None]], axis=2)
        else:
            out[y0:y1] = rgb

    out_img = Image.fromarray(out, mode="RGBA" if alpha_on else "RGB")

    buf = io.BytesIO()
    if fmt in ("jpg", "jpeg"):
        out_img.save(buf, format="JPEG", quality=95, subsampling=0)
        ctype = "image/jpeg"
    else:
        out_img.save(buf, format="PNG", optimize=True)
        ctype = "image/png"
    body = buf.getvalue()

    etag = hashlib.md5(
        f"{w}x{h}:{seed}:{nstops}:{hue_span}:{vibrance}:{bright}:{gamma}:{invert}:{strength}:{alpha_on}:{fmt}".encode()
    ).hexdigest()
    resp = HttpResponse(b"" if request.method == "HEAD" else body, content_type=ctype)
    resp["Content-Length"] = str(len(body))
    resp["ETag"] = etag
    resp["Cache-Control"] = "public, max-age=3600"
    return resp

# ---- Config
MAX_TEXT_LEN = 2000
MAX_DESCRIPTORS = 16
OPENAI_MODEL_DEFAULT = getattr(settings, "OPENAI_TEXT_MODEL", "gpt-4.1-mini")
PLANETARY_FALLBACK_ENABLED = getattr(settings, "PLANETARY_FALLBACK_ENABLED", True)

# ---- Sanitizers
_CTRL = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f]+")

def _clean_text(s: str, max_len: int = MAX_TEXT_LEN) -> str:
    if not isinstance(s, str):
        s = str(s or "")
    s = _CTRL.sub("", s)
    s = " ".join(s.split())
    return s[:max_len].strip()

def _clean_descriptors(items) -> list:
    if not items:
        return []
    out, seen = [], set()
    for x in items:
        t = _clean_text(str(x), max_len=48)
        if t and t.lower() not in seen:
            seen.add(t.lower()); out.append(t)
        if len(out) >= MAX_DESCRIPTORS:
            break
    return out

# ---- Deterministic geo from text (uniform on sphere)
def _geo_from_text(personal_account: str, salt: str = "") -> Tuple[float, float, str]:
    msg = (personal_account or "").encode("utf-8")
    key = (salt or "").encode("utf-8")
    h = hashlib.blake2b(msg, digest_size=16, key=key).digest()
    a = int.from_bytes(h[:8], "big"); b = int.from_bytes(h[8:], "big")
    u = (a / 2**64) * 2.0 - 1.0         # [-1, 1]
    v = b / 2**64                        # [0, 1)
    lat = degrees(asin(max(-1.0, min(1.0, u))))     # [-90, 90]
    lon = (v * 360.0) - 180.0                        # [-180, 180)
    return lat, lon, h.hex()[:16]

# ---- Local fallback generator (NEVER echoes inputs)
def _local_planetary_view(
    personal_account: str,
    descriptors: Sequence[str],
    lat: float,
    lon: float,
) -> str:
    """
    Standalone paragraph; does not quote/mention the submitted text, keywords, or any country.
    Uses a deterministic seed derived from personal_account for style variety.
    """
    seed = int(hashlib.blake2b(personal_account.encode(), digest_size=8).hexdigest(), 16)
    rng = random.Random(seed)

    # Word pools (influenced only by RNG; no direct echoing)
    lenses = ["personal", "ecological", "intergenerational", "socio-economic", "collective"]
    verbs = ["ripple", "braid", "converge", "echo", "spiral", "interlace"]
    motifs = ["smoke", "dust", "heat", "noise", "flood", "drought", "metal", "ash", "salt"]
    tones  = ["tender", "grounded", "clear-eyed", "resolute", "steadfast"]

    lens = ", ".join(rng.sample(lenses, 2))
    verb = rng.choice(verbs)
    motif = ", ".join(rng.sample(motifs, 2))
    tone = rng.choice(tones)

    # Build a 100–160 word-ish paragraph without referring to inputs
    body = (
        f"Pain is not solitary. It {verb}s across bodies and biomes, tying {lens} layers together. "
        f"In the currents of daily life, signals condense as {motif}, and pressure gathers across air, water, soil, and supply chains. "
        f"What seems isolated is braided with ecosystems, labor, and time—an ache that travels through households, rivers, forests, and markets. "
        f"The work is to witness and repair: trace the contours, render them visible, and act in kinship with more-than-human worlds. "
        f"Attunement is a practice: map, name, convene, and transform. ({tone}) "
        f"#SharedPain #PlanetaryHealth #PPPMap"
    )
    return " ".join(body.split())[:1000]

@api_view(["POST"])
@permission_classes([AllowAny])
def planetary_pain_text(request):
    """
    POST JSON:
      {
        "personal_account": "...",            # required (used for seeding only)
        "pain_descriptors": ["anxiety"...],   # optional (not echoed)
        "country": "..."                      # optional (ignored in output)
      }
    Always returns a standalone paragraph that does not reference inputs.
    """
    # Parse input
    personal_account = _clean_text(request.data.get("personal_account", ""))
    if not personal_account:
        return JsonResponse({"error": "personal_account is required"}, status=400)
    pain_descriptors = _clean_descriptors(request.data.get("pain_descriptors"))
    # country may be provided but MUST NOT be mentioned in output
    _ = _clean_text(request.data.get("country", ""), max_len=64)  # parsed then ignored

    # Deterministic geo anchor
    salt = getattr(settings, "PAIN_GEO_SALT", (settings.SECRET_KEY or "")[:16])
    lat, lon, seed_id = _geo_from_text(personal_account, salt=salt)

    # Bumpmap link
    try:
        bumpmap_base = request.build_absolute_uri(reverse("bumpmap"))
    except Exception:
        bumpmap_base = request.build_absolute_uri("/api/bumpmap/")
    bumpmap_url = f"{bumpmap_base}?w=2048&h=1024&lat={lat:.6f}&lon={lon:.6f}&sigma=20"

    # Try OpenAI (without sending the personal text or country)
    api_key = getattr(settings, "OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
    model = OPENAI_MODEL_DEFAULT

    if OpenAI and api_key:
        try:
            client = OpenAI(api_key=api_key, timeout=15)

            # We provide only internal anchors; we DO NOT include user text or country.
            # We also instruct the model explicitly to avoid referencing inputs.
            instructions = (
                "You are an artist-ecologist voice for the PPP (Personal–Planetary–Pain) map. "
                "Write ONE standalone paragraph (100–160 words) that connects personal, ecological, "
                "and socio-economic dimensions of shared pain. Keep it grounded, sensory, constructive. "
                "Never address the reader; never mention prompts, inputs, coordinates, countries, or keywords. "
                "Do not quote or paraphrase any provided text. End with 3 short hashtags."
            )

            # Lightweight internal cues (seed + hidden coords) are given ONLY as internal guidance.
            # The model is told not to reveal or reference them.
            internal_cues = (
                f"Internal anchors (do NOT mention in output): "
                f"seed={seed_id}, lat={lat:.4f}, lon={lon:.4f}."
            )

            # Optionally nudge tone using hashed choices from descriptors (without listing them)
            rng = random.Random(int(hashlib.blake2b("|".join(pain_descriptors).encode(), digest_size=8).hexdigest(), 16) if pain_descriptors else 0)
            tone_words = ["tender", "grounded", "clear-eyed", "resolute", "steadfast"]
            tone_hint = rng.choice(tone_words) if pain_descriptors else "grounded"

            user_input = (
                f"{internal_cues}\n"
                f"Desired tone (do NOT mention explicitly): {tone_hint}.\n"
                "Task: Produce the paragraph now."
            )

            resp = client.responses.create(
                model=model,
                instructions=instructions,
                input=user_input,
                max_output_tokens=500
            )
            text = (resp.output_text or "").strip()
            if not text:
                raise RuntimeError("Empty response text from OpenAI")

            # best-effort post-filter: strip accidental mentions
            text = re.sub(r"\b(Chile|Argentina|USA|United States|China|India|Europe|Africa|Asia|Australia)\b", " ", text)
            text = re.sub(r"\b(descriptors?|keywords?|prompt|input|coordinates?|latitude|longitude)\b", " ", text, flags=re.I)
            text = " ".join(text.split())

            return JsonResponse({
                "planetary_view": text,
                "source": "openai",
                "model": model,
                "deterministic_seed": seed_id,
                "lat": round(lat, 6),
                "lon": round(lon, 6),
                "bumpmap_url": bumpmap_url
            }, status=200)

        except Exception as e:
            if PLANETARY_FALLBACK_ENABLED:
                text = _local_planetary_view(
                    personal_account=personal_account,
                    descriptors=pain_descriptors,
                    lat=lat,
                    lon=lon,
                )
                resp = JsonResponse({
                    "planetary_view": text,
                    "source": "local_fallback",
                    "error": f"openai_error:{e.__class__.__name__}",
                    "deterministic_seed": seed_id,
                    "lat": round(lat, 6),
                    "lon": round(lon, 6),
                    "bumpmap_url": bumpmap_url
                }, status=200)
                resp["Retry-After"] = "60"
                return resp
            return JsonResponse({"error": "OpenAI unavailable"}, status=502)

    # No OpenAI configured → fallback
    text = _local_planetary_view(
        personal_account=personal_account,
        descriptors=pain_descriptors,
        lat=lat,
        lon=lon,
    )
    return JsonResponse({
        "planetary_view": text,
        "source": "local_fallback",
        "deterministic_seed": seed_id,
        "lat": round(lat, 6),
        "lon": round(lon, 6),
        "bumpmap_url": bumpmap_url
    }, status=200)
