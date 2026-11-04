"""AquaSentinel Dashboard Demonstration Scenario Generator.

Generates realistic SAR backscatter profiles, flood masks, and geospatial
bounding coordinates for out-of-the-box dashboard simulation and evaluation.
"""

from typing import Dict, Any, Tuple
import numpy as np


def generate_synthetic_sar_scene(
    scenario: str = "Sylhet Basin Monsoon Flood",
    size: Tuple[int, int] = (200, 200),
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Generate synthetic pre-event and post-event SAR backscatter imagery,
    ground-truth flood mask, and geographic coordinate bounds.
    """
    rng = np.random.default_rng(seed)
    h, w = size

    # Create base terrain roughness (SAR backscatter in dB, typically -25 to -5 dB)
    y, x = np.mgrid[0:h, 0:w]

    # Define coordinate bounds based on scenario
    if "Sylhet" in scenario or "South Asia" in scenario:
        lat_range = (24.75, 25.15)
        lon_range = (91.65, 92.10)
        center_lat, center_lon = 24.95, 91.88
        region_desc = "Surma-Kushiyara River Basin, Sylhet Division"
    else:
        lat_range = (44.10, 44.45)
        lon_range = (11.85, 12.35)
        center_lat, center_lon = 44.28, 12.10
        region_desc = "Emilia-Romagna River Basin, Northern Italy"

    # Pre-event VV: mostly land with a thin winding river
    river_path = (w // 2) + np.sin(y / 15.0) * 25.0
    river_mask = np.abs(x - river_path) < 4

    # Water has very low SAR backscatter (specular reflection, around -20 to -24 dB)
    # Land has higher backscatter (around -12 to -8 dB)
    pre_vv = rng.normal(loc=-11.0, scale=2.5, size=(h, w))
    pre_vv[river_mask] = rng.normal(loc=-22.0, scale=1.5, size=np.sum(river_mask))

    pre_vh = pre_vv - rng.normal(loc=6.5, scale=1.0, size=(h, w))

    # Post-event flood: river breaks banks and floods adjacent lowlands
    flood_extent = (
        (np.abs(x - river_path) < (22 + np.sin(y / 10.0) * 12)) &
        (y > 35) & (y < 175)
    )
    # Add localized depression inundation pockets
    depressions = ((x - 50)**2 + (y - 70)**2 < 20**2) | ((x - 145)**2 + (y - 130)**2 < 25**2)
    flood_extent = flood_extent | depressions

    post_vv = pre_vv.copy()
    post_vh = pre_vh.copy()

    # Inundated areas now reflect specularly (dark in SAR)
    post_vv[flood_extent] = rng.normal(loc=-21.5, scale=1.8, size=np.sum(flood_extent))
    post_vh[flood_extent] = post_vv[flood_extent] - rng.normal(loc=7.0, scale=1.0, size=np.sum(flood_extent))

    # Binary flood prediction mask
    flood_mask = np.zeros((h, w), dtype=np.uint8)
    flood_mask[flood_extent] = 255

    return {
        "scenario": scenario,
        "region_desc": region_desc,
        "center_lat": center_lat,
        "center_lon": center_lon,
        "lat_range": lat_range,
        "lon_range": lon_range,
        "pre_vv": pre_vv,
        "pre_vh": pre_vh,
        "post_vv": post_vv,
        "post_vh": post_vh,
        "flood_mask": flood_mask,
        "pixel_resolution_m": 20.0,
    }
