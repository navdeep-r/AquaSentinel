"""AquaSentinel GeoJSON Vectorization & Hazard Polygon Exporter.

Converts SAR flood raster predictions into GeoJSON vector polygons with
inundation area metrics, centroid alerts, and GIS-ready metadata.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from shapely.geometry import shape, mapping, Polygon, MultiPolygon
from shapely.ops import unary_union

try:
    import rasterio
    import rasterio.features
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


def mask_to_geojson_features(
    mask: np.ndarray,
    transform: Any = None,
    crs: str = "EPSG:4326",
    min_area_hectares: float = 0.2,
    pixel_res_m: float = 10.0,
) -> Dict[str, Any]:
    """
    Convert a 2D binary numpy array / raster mask into a GeoJSON FeatureCollection.

    Args:
        mask: 2D binary array (values > 0 represent flood extent).
        transform: Affine transform from raster coordinates to geographic/projected CRS.
        crs: Coordinate reference system string.
        min_area_hectares: Discard speckle noise clusters smaller than this area.
        pixel_res_m: Approximate ground resolution per pixel in meters.

    Returns:
        GeoJSON FeatureCollection dict.
    """
    binary_mask = (mask > 0).astype(np.uint8)
    features: List[Dict[str, Any]] = []

    if HAS_RASTERIO and transform is not None:
        shapes_gen = rasterio.features.shapes(binary_mask, mask=(binary_mask == 1), transform=transform)
        raw_polys = []
        for geom_dict, val in shapes_gen:
            if val == 1:
                poly = shape(geom_dict)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if not poly.is_empty:
                    raw_polys.append(poly)
    else:
        # Fallback coordinate generation if transform not supplied
        h, w = mask.shape
        raw_polys = []
        # Simple box approximation for testing without transform
        y_idxs, x_idxs = np.where(binary_mask == 1)
        if len(y_idxs) > 0:
            min_x, max_x = float(np.min(x_idxs)), float(np.max(x_idxs))
            min_y, max_y = float(np.min(y_idxs)), float(np.max(y_idxs))
            raw_polys.append(Polygon([(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]))

    # Calculate area and filter
    sq_m_per_pixel = pixel_res_m * pixel_res_m
    sq_m_per_ha = 10_000.0

    poly_id = 1
    for poly in raw_polys:
        # Approximate area in hectares
        approx_pixels = poly.area if transform is None else (poly.area / (abs(transform.a * transform.e) if transform else 1.0))
        area_ha = (approx_pixels * sq_m_per_pixel) / sq_m_per_ha
        area_sq_km = area_ha / 100.0

        if area_ha < min_area_hectares:
            continue

        centroid = poly.centroid
        alert_priority = "High" if area_ha > 15.0 else ("Medium" if area_ha > 3.0 else "Low")

        feature = {
            "type": "Feature",
            "id": poly_id,
            "geometry": mapping(poly),
            "properties": {
                "flood_polygon_id": poly_id,
                "inundated_area_hectares": round(area_ha, 2),
                "inundated_area_sq_km": round(area_sq_km, 3),
                "centroid_x": round(centroid.x, 6),
                "centroid_y": round(centroid.y, 6),
                "hazard_priority": alert_priority,
                "detection_engine": "AquaSentinel-SAR",
            },
        }
        features.append(feature)
        poly_id += 1

    feature_collection = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {"name": str(crs)},
        },
        "features": features,
    }
    return feature_collection


def raster_to_geojson_file(
    input_raster: Union[str, Path],
    output_geojson: Union[str, Path],
    min_area_hectares: float = 0.2,
) -> Dict[str, Any]:
    """Convert an input flood GeoTIFF directly to a GeoJSON file."""
    if not HAS_RASTERIO:
        raise ImportError("rasterio is required to convert raster files to GeoJSON.")

    input_path = Path(input_raster)
    out_path = Path(output_geojson)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(input_path) as src:
        mask = src.read(1)
        transform = src.transform
        crs = str(src.crs)
        res_m = (abs(src.transform.a) + abs(src.transform.e)) / 2.0
        if src.crs and not src.crs.is_projected:
            res_m *= 111320.0

    fc = mask_to_geojson_features(
        mask=mask,
        transform=transform,
        crs=crs,
        min_area_hectares=min_area_hectares,
        pixel_res_m=res_m,
    )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(fc, f, indent=2)

    print(f"Exported {len(fc['features'])} flood polygon features to {out_path}")
    return fc


def main():
    parser = argparse.ArgumentParser(description="Convert AquaSentinel flood raster to GeoJSON vector polygons.")
    parser.add_argument("--input", type=str, required=True, help="Path to input flood prediction GeoTIFF.")
    parser.add_argument("--output", type=str, required=True, help="Path for output GeoJSON file.")
    parser.add_argument("--min_area_ha", type=float, default=0.2, help="Minimum polygon area in hectares to retain.")
    args = parser.parse_args()

    raster_to_geojson_file(args.input, args.output, min_area_hectares=args.min_area_ha)


if __name__ == "__main__":
    main()
