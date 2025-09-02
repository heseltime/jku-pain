🥐 **Johannes Kepler University MED Faculty and Ludwig Boltzmann Institutes: Transforming Medicine through AI and Art** 🥐

> [!NOTE]  
> **P.ersonal A.nd I.n N.ature** Project Server 

# Documentation

Main endpoint: `/api/bumpmap/` (deployed on render: `https://pain-ix0y.onrender.com/api/bumpmap/`)

## Parameters

| Name        | Type  | Default | Range/Notes                                                                 |
|-------------|-------|---------|------------------------------------------------------------------------------|
| `w`         | int   | `8192`  | Image width (px). For world maps, use **2:1** aspect with `h = w/2`.        |
| `h`         | int   | `4096`  | Image height (px).                                                          |
| `lat`       | float | `50.0`  | Center latitude in **degrees** (−90..90).                                   |
| `lon`       | float | `10.0`  | Center longitude in **degrees** (−180..180).                                 |
| `sigma`     | float | `20.0`  | **Angular radius** in degrees for Gaussian falloff. Larger ⇒ wider bright area. |
| `scale`     | float | `1.0`   | Multiplier applied to `sigma`. Effective radius = `sigma * scale` (clamped 0.1..10). |
| `hard`      | bool  | `false` | `1,true,on` ⇒ binary mask (white inside, black outside). Otherwise grayscale. |
| `threshold` | float | `0.35`  | Only used when `hard=1`. Pixels with intensity ≥ threshold become white.     |
| `fmt`       | enum  | `png`   | `png`, `jpg`, `jpeg`.                                                        |

---

## Examples

**Default (8K grayscale around Europe)**
```bash
curl -L -o europe_bump_8k.png \
  "https://pain-ix0y.onrender.com/api/bumpmap/"
```

**Scale up crater ~2× (effective radius 40°)**
```bash
curl -L -o bump_big.png \
  "https://pain-ix0y.onrender.com/api/bumpmap/?sigma=20&scale=2"
```

**Binary mask with custom threshold**
```bash
curl -L -o europe_mask.png \
  "https://pain-ix0y.onrender.com/api/bumpmap/?hard=1&threshold=0.3"
```

**JPEG instead of PNG**
```bash
curl -L -o bump.jpg \
  "https://pain-ix0y.onrender.com/api/bumpmap/?fmt=jpg"
```

**Quick health check (headers only)**
```bash
curl -I "https://pain-ix0y.onrender.com/api/bumpmap/?w=1024&h=512&sigma=20"
```