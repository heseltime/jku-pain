from django.shortcuts import render

import io, hashlib
import numpy as np
from PIL import Image, ImageFilter
from django.http import HttpResponse
from rest_framework.decorators import api_view

def _fbm_noise(w: int, h: int, seed=None, octaves: int = 5) -> Image.Image:
    """
    Simple fractal noise: sums several upsampled random fields for a smooth bump map.
    Returns an 8-bit grayscale Pillow Image (mode 'L').
    """
    rng = np.random.default_rng(None if seed is None else int(seed))
    acc = np.zeros((h, w), dtype=np.float32)
    amp, amp_sum = 1.0, 0.0

    # Start from coarse grids and go finer
    for i in range(octaves):
        # base grid size grows coarser as i increases; tweak 8 to shape smoothness
        gh = max(1, h // (2 ** i * 8))
        gw = max(1, w // (2 ** i * 8))
        base = rng.random((gh, gw)).astype(np.float32)

        # Upsample to (w,h) and blend
        layer = Image.fromarray((base * 255).astype('uint8'), mode='L').resize((w, h), Image.BILINEAR)
        acc += (np.asarray(layer, dtype=np.float32) / 255.0) * amp
        amp_sum += amp
        amp *= 0.5

    acc /= max(amp_sum, 1e-6)
    acc = np.clip(acc, 0.0, 1.0)

    # Optional light blur to soften pixel grid artifacts
    img = Image.fromarray((acc * 255).astype('uint8'), mode='L').filter(ImageFilter.GaussianBlur(radius=1))
    return img

@api_view(['GET'])
def bumpmap(request):
    # Query params with sensible defaults for a world-map aspect
    w = int(request.GET.get('w', 2048))
    h = int(request.GET.get('h', 1024))
    seed = request.GET.get('seed')  # optional, deterministic output if provided

    # Generate image
    img = _fbm_noise(w, h, seed=seed, octaves=5)

    # Serialize to PNG
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    png = buf.getvalue()

    # Basic ETag for client caching
    etag = hashlib.md5(f"{w}x{h}:{seed}".encode('utf-8')).hexdigest()
    resp = HttpResponse(png, content_type='image/png')
    resp['ETag'] = etag
    resp['Cache-Control'] = 'public, max-age=3600'
    return resp
