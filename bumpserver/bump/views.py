import io
import hashlib
from math import pi
import numpy as np
from PIL import Image
from django.http import HttpResponse
from rest_framework.decorators import api_view

# --- helpers ---------------------------------------------------------------

def _parse_bool(val, default=False):
    if val is None:
        return default
    return str(val).lower() in ("1", "true", "t", "yes", "y", "on")

def _equirectangular_gaussian(
    w: int,
    h: int,
    lat0_deg: float = 50.0,   # ~Central Europe
    lon0_deg: float = 10.0,
    sigma_deg: float = 20.0,  # spherical Gaussian radius
    hard: bool = False,
    threshold: float = 0.35   # only used if hard=True
) -> Image.Image:
    """
    Create an 8-bit grayscale image in equirectangular projection where the
    intensity is a spherical Gaussian around (lat0, lon0). With hard=True, it
    returns a binary mask: white inside, black outside.
    """
    lat0 = np.deg2rad(lat0_deg)
    lon0 = np.deg2rad(lon0_deg)

    # grid in radians: lon ∈ [-π, π), lat ∈ [π/2, -π/2]
    lon = np.linspace(-pi, pi, w, endpoint=False)
    lat = np.linspace(pi/2, -pi/2, h)
    Lon, Lat = np.meshgrid(lon, lat)

    # great-circle distance (haversine)
    dlat = Lat - lat0
    dlon = (Lon - lon0 + pi) % (2 * pi) - pi  # wrap to [-π, π]
    a = np.sin(dlat/2)**2 + np.cos(Lat) * np.cos(lat0) * np.sin(dlon/2)**2
    gc = 2 * np.arcsin(np.sqrt(a))  # radians on unit sphere

    sigma = np.deg2rad(sigma_deg)
    intensity = np.exp(-(gc**2) / (2 * sigma**2))  # 0..1

    if hard:
        arr = (intensity >= threshold).astype(np.uint8) * 255
    else:
        arr = np.clip(intensity * 255.0, 0, 255).astype(np.uint8)

    return Image.fromarray(arr, mode="L")

# --- endpoint --------------------------------------------------------------

@api_view(["GET"])
def bumpmap(request):
    """
    GET /api/bumpmap/?
        w=8192&h=4096
        lat=50&lon=10
        sigma=20         # degrees (soft falloff radius)
        hard=0|1         # 1 => binary mask
        threshold=0.35   # for hard=1
        fmt=png|jpg
    """
    # dimensions (defaults to 8K 2:1)
    w = int(request.GET.get("w", 8192))
    h = int(request.GET.get("h", 4096))

    # center & shape
    lat = float(request.GET.get("lat", 50.0))
    lon = float(request.GET.get("lon", 10.0))
    sigma = float(request.GET.get("sigma", 20.0))
    hard = _parse_bool(request.GET.get("hard"), default=False)
    threshold = float(request.GET.get("threshold", 0.35))
    fmt = (request.GET.get("fmt", "png") or "png").lower()
    if fmt not in ("png", "jpg", "jpeg"):
        fmt = "png"

    img = _equirectangular_gaussian(
        w=w, h=h, lat0_deg=lat, lon0_deg=lon,
        sigma_deg=sigma, hard=hard, threshold=threshold
    )

    # serialize
    buf = io.BytesIO()
    if fmt in ("jpg", "jpeg"):
        img.save(buf, format="JPEG", quality=95, subsampling=0)
        content_type = "image/jpeg"
    else:
        img.save(buf, format="PNG", optimize=True)
        content_type = "image/png"
    body = buf.getvalue()

    # response with cache hints
    etag_src = f"{w}x{h}:{lat}:{lon}:{sigma}:{hard}:{threshold}:{fmt}"
    resp = HttpResponse(body, content_type=content_type)
    resp["Content-Length"] = str(len(body))
    resp["ETag"] = hashlib.md5(etag_src.encode("utf-8")).hexdigest()
    resp["Cache-Control"] = "public, max-age=3600"
    return resp
