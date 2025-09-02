# bump/views.py
import io, os, hashlib
from math import pi, asin, degrees
import numpy as np
from PIL import Image
from django.http import HttpResponse, JsonResponse
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny

from django.conf import settings
import colorsys

EARTH_SRC_PATH = os.getenv(
    "EARTH_TEXTURE_SRC",
    str((settings.BASE_DIR / "assets" / "8k_earth_daymap_greyscale.jpg").resolve())
)

def _parse_bool(v, default=False):
    return str(v).lower() in ("1","true","t","yes","y","on") if v is not None else default

def _equirectangular_gaussian_memsafe(
    w: int, h: int,
    lat0_deg: float = 50.0, lon0_deg: float = 10.0,
    sigma_deg: float = 20.0, hard: bool = False, threshold: float = 0.35
):
    # Longitudes precomputed once (float32)
    lon = np.linspace(-pi, pi, w, endpoint=False, dtype=np.float32)
    lon0 = np.float32(np.deg2rad(lon0_deg))
    dlon = (lon - lon0 + np.float32(pi)) % np.float32(2*pi) - np.float32(pi)
    sin2_dlon = np.sin(dlon / 2.0) ** 2   # float32

    lat0 = np.float32(np.deg2rad(lat0_deg))
    cos_lat0 = np.cos(lat0)               # float32
    sigma = np.float32(np.deg2rad(sigma_deg))

    out = np.empty((h, w), dtype=np.uint8)
    lats = np.linspace(pi/2, -pi/2, h, dtype=np.float32)

    for i in range(h):
        lat = lats[i]
        dlat = lat - lat0
        sin2_dlat = np.sin(dlat / 2.0) ** 2
        a = sin2_dlat + (np.cos(lat) * cos_lat0) * sin2_dlon
        gc = 2.0 * np.arcsin(np.sqrt(a))             # radians
        intensity = np.exp(-(gc * gc) / (2.0 * sigma * sigma))  # 0..1

        if hard:
            row = (intensity >= threshold).astype(np.uint8) * 255
        else:
            row = np.clip(intensity * 255.0, 0, 255).astype(np.uint8)
        out[i] = row

    return Image.fromarray(out, mode='L')

@api_view(["GET", "HEAD", "OPTIONS"])
def bumpmap(request):
    w = int(request.GET.get("w", 8192))
    h = int(request.GET.get("h", 4096))
    lat = float(request.GET.get("lat", 50.0))
    lon = float(request.GET.get("lon", 10.0))
    sigma = float(request.GET.get("sigma", 20.0))  # base radius in degrees

    # NEW: optional scale multiplier for the "crater" size
    try:
        scale = float(request.GET.get("scale", 1.0))
    except (TypeError, ValueError):
        scale = 1.0
    # keep it positive and within a reasonable range
    if scale <= 0 or not np.isfinite(scale):
        scale = 1.0
    scale = max(0.1, min(scale, 10.0))  # allow 0.1x .. 10x

    sigma_eff = sigma * scale  # effective radius in degrees

    hard = _parse_bool(request.GET.get("hard"), False)
    threshold = float(request.GET.get("threshold", 0.35))
    fmt = (request.GET.get("fmt", "png") or "png").lower()

    # use effective sigma
    img = _equirectangular_gaussian_memsafe(w, h, lat, lon, sigma_eff, hard, threshold)

    buf = io.BytesIO()
    if fmt in ("jpg", "jpeg"):
        img.save(buf, format="JPEG", quality=95, subsampling=0)
        ctype = "image/jpeg"
    else:
        img.save(buf, format="PNG", optimize=True)
        ctype = "image/png"
    body = buf.getvalue()

    # include sigma_eff in the cache key
    etag = hashlib.md5(f"{w}x{h}:{lat}:{lon}:{sigma_eff}:{hard}:{threshold}:{fmt}".encode()).hexdigest()
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

def _geo_from_text(personal_account: str, salt: str = ""):
    """
    Deterministically map arbitrary text -> (lat, lon) that is uniform on the sphere.
    Uses a keyed BLAKE2b hash for stability and privacy.
    """
    key = (salt or "").encode("utf-8")
    h = hashlib.blake2b(personal_account.encode("utf-8"), digest_size=16, key=key).digest()
    a = int.from_bytes(h[:8], "big", signed=False)
    b = int.from_bytes(h[8:], "big", signed=False)

    u = (a / 2**64) * 2.0 - 1.0         # uniform in [-1, 1]
    v = b / 2**64                        # uniform in [0, 1)

    lat = degrees(asin(max(-1.0, min(1.0, u))))  # [-90, 90]
    lon = (v * 360.0) - 180.0                     # [-180, 180)

    return lat, lon, h.hex()[:16]  # short seed id for debugging

@api_view(["POST"])
@permission_classes([AllowAny])
def locate_pain(request):
    """
    POST JSON: { "personal_account": "..." }
    -> { "lat": ..., "lon": ..., "seed": "...", "bumpmap_url": "..." }
    """
    personal_account = (request.data.get("personal_account") or "").strip()
    if not personal_account:
        return JsonResponse({"error": "personal_account is required"}, status=400)

    # Optional instance-specific salt so the mapping can't be trivially reversed elsewhere
    salt = getattr(settings, "PAIN_GEO_SALT", settings.SECRET_KEY[:16])

    lat, lon, seed_id = _geo_from_text(personal_account, salt=salt)

    # Handy link to your existing bumpmap endpoint for immediate visualization
    bumpmap_url = f"/bumpmap?w=2048&h=1024&lat={lat:.6f}&lon={lon:.6f}&sigma=20"

    return JsonResponse({
        "lat": round(lat, 6),
        "lon": round(lon, 6),
        "deterministic": True,
        "seed": seed_id,
        "bumpmap_url": bumpmap_url,
        "method": "hash->uniform-sphere(asin)"
    }, status=200)