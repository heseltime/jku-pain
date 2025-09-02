🥐 **Johannes Kepler University MED Faculty and Ludwig Boltzmann Institutes: Transforming Medicine through AI and Art** 🥐

> [!NOTE]  
> **P.ersonal A.nd I.n N.ature** Project Server 

# Documentation

## bumpmap

Main endpoint: `/api/bumpmap/` (deployed on render: `https://pain-ix0y.onrender.com/api/bumpmap/`) For generating a dent in the unshapely pain earth body: this should ultimately **reflect a global state of pain** based on user questionaire submissions. (This leads into another endpoint, where these questionaires are submitted to, providing feedback to the client in terms of the pain dimensions of this project.)

### Input

None/select Linz, AT as the epicenter (of pain).

### Parameters

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

### Examples

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

### Output

Black and white bumpmap image, scalable, reflective of global state based on form inputs.


## cloudmap

Three pain dimensions physiological and physical, emotional and socio-economic (randomly mocked before including data sources, see miro board). Matches the five sounds of the Traditional Chinese Medicine elements.

Alternative: Clouds stay static/the client uses its own file.


## earthtexture

Might take on the color from the clouds, i.e. a function (Mike maybe) to map data to a faded, multi-hue rectangle. Generates a soft, multi-hue CMY heat-map version of the grayscale Earth day texture (cyan → magenta → yellow with gentle dark/bright knees).

### Input

None required (uses the built-in grayscale basemap). A data-driven version can later swap the LUT index source from the map luminance to data values.

### Parameters

| Name       | Type   | Default | Range/Notes                                                                                          |
|------------|--------|---------|--------------------------------------------------------------------------------------------------------|
| `w`        | int    | `8192`  | Output width (px). For world maps use **2:1** aspect ⇒ set `h = w/2`.                                 |
| `h`        | int    | `4096`  | Output height (px).                                                                                    |
| `pastel`   | float  | `0.35`  | `0..1`. Mixes CMY with white. **Lower = more saturated**, higher = softer/faded.                      |
| `gamma`    | float  | `1.0`   | Gamma on the **grayscale** before the LUT. `<1` brightens mids; `>1` darkens.                         |
| `invert`   | bool   | `0`     | `1,true,t,yes,y,on` ⇒ invert grayscale before LUT (swaps low/high colors).                            |
| `strength` | float  | `0.35`  | `0..1`. Blend toward color. `0` = pure grayscale, `1` = full color. Try `0.75` for vivid output.      |
| `alpha`    | bool   | `0`     | `1,true,t,yes,y,on` ⇒ include alpha channel set to the **original grayscale** (PNG RGBA output).      |
| `fmt`      | enum   | `png`   | Output format: `png`, `jpg`, `jpeg`.                                                                   |

> Deprecated/ignored (kept for compatibility): `seed`, `nstops`, `hue_span`, `vibrance`, `bright`.

### Examples

```bash
curl -L -o earth_cmy_8k.png \
  "https://pain-ix0y.onrender.com/api/earthtexture/"
```

```bash
curl -L -o earth_cmy_vivid.png \
  "https://pain-ix0y.onrender.com/api/earthtexture/?strength=0.75&pastel=0.2"
```

```bash
curl -L -o earth_cmy_soft.png \
  "https://pain-ix0y.onrender.com/api/earthtexture/?strength=0.5&pastel=0.5"
```

### Output

Multi-color hue version of the earth map. Generates a soft, multi-hue CMY heat-map version of the grayscale Earth day texture (cyan → magenta → yellow with gentle dark/bright knees).


## locate_pain

### Input

Text pain prompt.

### POST Parameters

`personal_account` in JSON schema.

### Examples

```bash
curl -X POST http://localhost:8000/api/locate-pain \
  -H "Content-Type: application/json" \
  -d '{"personal_account":"I feel a tightness in my chest when thinking about wildfires."}'
```

### Output

Location coordinates mainly:

```bash
{
    "lat": -68.269179, 
    "lon": -111.796729, 
    "deterministic": true, 
    "seed": "0918b4a9b7469719", 
    "bumpmap_url": "/bumpmap?w=2048&h=1024&lat=-68.269179&lon=-111.796729&sigma=20", 
    "method": "hash->uniform-sphere(asin)"
}
```