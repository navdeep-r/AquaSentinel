"""AquaSentinel Geospatial Flood Risk & Impact Assessment Engine.

Computes physical inundation areas, population and agricultural exposures,
severity ratings, and automated disaster situation reports (SitReps) from
SAR-derived flood masks.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


class SeverityLevel(str, Enum):
    """Standard disaster emergency flood severity classifications."""
    LOW = "Level 1: Advisory (Localized Inundation)"
    MODERATE = "Level 2: Warning (Moderate Flooding)"
    HIGH = "Level 3: Severe (High Inundation / Evacuation Alert)"
    CRITICAL = "Level 4: Catastrophic (Critical Emergency)"


@dataclass
class FloodImpactMetrics:
    """Comprehensive flood impact metrics summary."""
    flooded_pixels: int
    total_valid_pixels: int
    pixel_resolution_m: float
    flooded_area_sq_m: float
    flooded_area_hectares: float
    flooded_area_sq_km: float
    inundation_percentage: float
    severity_level: SeverityLevel
    estimated_affected_population: int
    impacted_cropland_hectares: float
    impacted_urban_area_hectares: float
    infrastructure_risk_index: float  # Scale 0.0 - 100.0
    recommended_actions: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to serializable dictionary."""
        return {
            "timestamp": self.timestamp,
            "flooded_pixels": self.flooded_pixels,
            "total_valid_pixels": self.total_valid_pixels,
            "pixel_resolution_meters": self.pixel_resolution_m,
            "flooded_area_sq_km": round(self.flooded_area_sq_km, 3),
            "flooded_area_hectares": round(self.flooded_area_hectares, 2),
            "inundation_percentage": round(self.inundation_percentage, 2),
            "severity_level": self.severity_level.value,
            "estimated_affected_population": self.estimated_affected_population,
            "impacted_cropland_hectares": round(self.impacted_cropland_hectares, 2),
            "impacted_urban_area_hectares": round(self.impacted_urban_area_hectares, 2),
            "infrastructure_risk_index": round(self.infrastructure_risk_index, 1),
            "recommended_actions": self.recommended_actions,
        }


class FloodRiskAnalyzer:
    """Analytical engine for evaluating flood extents and humanitarian exposure."""

    def __init__(
        self,
        default_pop_density_per_sq_km: float = 180.0,
        cropland_fraction_estimate: float = 0.55,
        urban_fraction_estimate: float = 0.18,
    ):
        """
        Initialize the analyzer with regional exposure coefficients.

        Args:
            default_pop_density_per_sq_km: Average population density in target basin.
            cropland_fraction_estimate: Estimated share of agricultural land in floodplain.
            urban_fraction_estimate: Estimated share of urban/settlement footprint.
        """
        self.default_pop_density = default_pop_density_per_sq_km
        self.cropland_fraction = cropland_fraction_estimate
        self.urban_fraction = urban_fraction_estimate

    def analyze_mask(
        self,
        mask: np.ndarray,
        pixel_resolution_m: float = 10.0,
        region_name: str = "Target AOI",
    ) -> FloodImpactMetrics:
        """
        Compute impact metrics directly from a 2D numpy binary array.

        Args:
            mask: Binary or uint8 array where values > 0 indicate flooded pixels.
            pixel_resolution_m: Ground sampling distance per pixel in meters.
            region_name: Descriptive name of the analyzed zone.

        Returns:
            FloodImpactMetrics instance with computed risk indicators.
        """
        valid_mask = ~np.isnan(mask)
        total_valid_pixels = int(np.sum(valid_mask))
        flooded_pixels = int(np.sum((mask > 0) & valid_mask))

        if total_valid_pixels == 0:
            inundation_pct = 0.0
        else:
            inundation_pct = (flooded_pixels / total_valid_pixels) * 100.0

        pixel_area_sq_m = pixel_resolution_m * pixel_resolution_m
        flooded_sq_m = flooded_pixels * pixel_area_sq_m
        flooded_ha = flooded_sq_m / 10_000.0
        flooded_sq_km = flooded_sq_m / 1_000_000.0

        # Assess severity level
        severity, actions = self._evaluate_severity(flooded_sq_km, inundation_pct)

        # Exposure estimations
        cropland_ha = flooded_ha * self.cropland_fraction
        urban_ha = flooded_ha * self.urban_fraction
        urban_sq_km = urban_ha / 100.0
        affected_pop = int(urban_sq_km * self.default_pop_density * 4.5)  # settlement concentration factor

        # Infrastructure risk index based on urban flood footprint & severity
        infra_risk = min(100.0, (flooded_sq_km * 0.4) + (inundation_pct * 1.5) + (affected_pop * 0.01))

        return FloodImpactMetrics(
            flooded_pixels=flooded_pixels,
            total_valid_pixels=total_valid_pixels,
            pixel_resolution_m=pixel_resolution_m,
            flooded_area_sq_m=flooded_sq_m,
            flooded_area_hectares=flooded_ha,
            flooded_area_sq_km=flooded_sq_km,
            inundation_percentage=inundation_pct,
            severity_level=severity,
            estimated_affected_population=affected_pop,
            impacted_cropland_hectares=cropland_ha,
            impacted_urban_area_hectares=urban_ha,
            infrastructure_risk_index=infra_risk,
            recommended_actions=actions,
        )

    def analyze_raster_file(
        self,
        raster_path: Union[str, Path],
        region_name: str = "Target AOI",
    ) -> Tuple[FloodImpactMetrics, Dict[str, Any]]:
        """
        Open a GeoTIFF flood prediction raster, calculate resolution and run assessment.
        """
        if not HAS_RASTERIO:
            raise ImportError("rasterio is required to analyze raster files directly.")

        path = Path(raster_path)
        with rasterio.open(path) as src:
            data = src.read(1)
            # Estimate pixel resolution in meters
            res_x, res_y = abs(src.transform.a), abs(src.transform.e)
            if src.crs and not src.crs.is_projected:
                # Approximate 1 degree ~ 111,320m at equator
                resolution_m = ((res_x + res_y) / 2.0) * 111320.0
            else:
                resolution_m = (res_x + res_y) / 2.0

            bounds = src.bounds
            crs_str = str(src.crs)

        geo_meta = {
            "file": str(path),
            "crs": crs_str,
            "bounds": {
                "left": bounds.left,
                "bottom": bounds.bottom,
                "right": bounds.right,
                "top": bounds.top,
            },
        }

        metrics = self.analyze_mask(data, pixel_resolution_m=resolution_m, region_name=region_name)
        return metrics, geo_meta

    def _evaluate_severity(self, flooded_sq_km: float, inundation_pct: float) -> Tuple[SeverityLevel, list[str]]:
        """Determine severity level and emergency response protocols."""
        if flooded_sq_km > 150 or inundation_pct > 25.0:
            severity = SeverityLevel.CRITICAL
            actions = [
                "Issue urgent Level 4 Red Flood Alert via NDMA/Civil Protection.",
                "Mobilize rapid SAR watercraft and rotary-wing search & rescue teams.",
                "Initiate immediate evacuation for low-lying settlements and riverine corridors.",
                "Establish emergency relief hubs and potable water distribution points.",
            ]
        elif flooded_sq_km > 40 or inundation_pct > 12.0:
            severity = SeverityLevel.HIGH
            actions = [
                "Activate Emergency Operations Center (EOC) and incident response standby.",
                "Prepare targeted evacuation routes for vulnerable downstream communities.",
                "Deploy temporary flood barriers and breach containment around key substations.",
                "Continuously monitor next Sentinel-1 / SAR overpass imagery.",
            ]
        elif flooded_sq_km > 10 or inundation_pct > 4.0:
            severity = SeverityLevel.MODERATE
            actions = [
                "Broadcast localized flood warnings to transport and agricultural authorities.",
                "Inspect culverts, drainage channels, and retention levees.",
                "Pre-position water pumps and medical supplies in affected municipalities.",
            ]
        else:
            severity = SeverityLevel.LOW
            actions = [
                "Maintain standard hydrologic and satellite observation protocol.",
                "Verify ground-truth sensor water levels across primary watercourses.",
            ]

        return severity, actions

    def generate_situation_report(
        self,
        metrics: FloodImpactMetrics,
        aoi_name: str = "Analyzed Basin",
        event_name: str = "Monsoon Surge / Extreme Precipitation Event",
    ) -> str:
        """Generate a formatted Markdown Situation Report (SitRep)."""
        report = f"""# 🌊 AquaSentinel Disaster Situation Report (SitRep)
**Incident:** {event_name}  
**Area of Interest (AOI):** {aoi_name}  
**Generated:** {metrics.timestamp}  
**Emergency Status:** **{metrics.severity_level.value}**  

---

### 1. Key Impact Indicators
| Metric | Value |
| :--- | :--- |
| **Total Inundated Area** | **{metrics.flooded_area_sq_km:,.2f} km²** ({metrics.flooded_area_hectares:,.1f} ha) |
| **Basin Inundation Ratio** | **{metrics.inundation_percentage:.2f}%** |
| **Est. Vulnerable Population** | **~{metrics.estimated_affected_population:,} persons** |
| **Agricultural Exposure** | **{metrics.impacted_cropland_hectares:,.1f} hectares** |
| **Urban Built-up Inundation** | **{metrics.impacted_urban_area_hectares:,.1f} hectares** |
| **Infrastructure Risk Index** | **{metrics.infrastructure_risk_index:.1f} / 100** |

---

### 2. Operational Directives & Recommended Actions
"""
        for i, action in enumerate(metrics.recommended_actions, 1):
            report += f"{i}. {action}\n"

        report += """
---
*Automated report compiled by AquaSentinel Synthetic Aperture Radar (SAR) Flood Intelligence Engine.*
"""
        return report
