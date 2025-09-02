# bump/views.py
import io, hashlib
from math import pi
import numpy as np
from PIL import Image
from django.http import HttpResponse
from rest_framework.decorators import api_view

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

