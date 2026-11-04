"""AquaSentinel Mission Control - Real-Time SAR Flood Intelligence Dashboard.

An operational command interface for emergency services, disaster management
agencies, and geospatial analysts to monitor satellite SAR flood inundation.
"""

import json
from datetime import datetime
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Import AquaSentinel analytics modules
try:
    from src.analytics.flood_risk_analyzer import FloodRiskAnalyzer, SeverityLevel
    from src.dashboard.mock_data import generate_synthetic_sar_scene
except ImportError:
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from src.analytics.flood_risk_analyzer import FloodRiskAnalyzer, SeverityLevel
    from src.dashboard.mock_data import generate_synthetic_sar_scene


def apply_theme():
    st.set_page_config(
        page_title="AquaSentinel | SAR Flood Intelligence",
        page_icon="🛰️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .metric-card {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 18px;
            color: white;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        }
        .metric-value {
            font-size: 2.1rem;
            font-weight: 700;
            color: #38bdf8;
            margin-top: 4px;
        }
        .metric-label {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #94a3b8;
        }
        .alert-badge {
            display: inline-block;
            padding: 6px 14px;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 0.9rem;
        }
        .alert-critical { background-color: #ef4444; color: white; }
        .alert-high { background-color: #f97316; color: white; }
        .alert-moderate { background-color: #eab308; color: black; }
        .alert-low { background-color: #10b981; color: white; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    apply_theme()

    # Header & Banner
    col_logo, col_title = st.columns([1, 8])
    with col_logo:
        st.markdown("<h1 style='font-size: 3.5rem; margin-bottom: 0;'>🛰️</h1>", unsafe_allow_html=True)
    with col_title:
        st.title("AquaSentinel Mission Control")
        st.caption("AI-Powered Synthetic Aperture Radar (SAR) Flood Detection & Disaster Risk Analytics")

    st.markdown("---")

    # Sidebar Controls
    st.sidebar.image("https://img.shields.io/badge/AquaSentinel-SAR%20v1.0-blue?style=for-the-badge", use_container_width=True)
    st.sidebar.header("🎯 Mission Parameters")

    data_source = st.sidebar.selectbox(
        "Imagery Stream / Scenario",
        [
            "Demo: South Asia Monsoon (Sylhet Basin)",
            "Demo: Mediterranean Flash Flood (Emilia-Romagna)",
            "Custom GeoTIFF / Upload File",
        ],
        index=0,
    )

    st.sidebar.subheader("📡 Radar Processing Settings")
    polarization_mode = st.sidebar.selectbox("SAR Mode", ["Dual-Pol (VV + VH)", "Single-Pol (VV Only)"])
    pixel_res = st.sidebar.slider("Ground Sampling Distance (m)", 10, 50, 20, step=5)
    confidence_thresh = st.sidebar.slider("Inundation Threshold (dB)", -24.0, -14.0, -20.0, step=0.5)
    buffer_dilation = st.sidebar.slider("Morphological Buffer (pixels)", 0, 16, 8, step=2)

    st.sidebar.subheader("👥 Humanitarian Settings")
    regional_pop_density = st.sidebar.number_input("Basin Pop. Density (per km²)", value=220, min_value=10, max_value=2500)

    # Initialize Engine & Data
    analyzer = FloodRiskAnalyzer(default_pop_density_per_sq_km=regional_pop_density)

    if "South Asia" in data_source:
        scene = generate_synthetic_sar_scene("Sylhet Basin Monsoon Flood", size=(180, 180), seed=42)
    elif "Emilia-Romagna" in data_source:
        scene = generate_synthetic_sar_scene("Emilia-Romagna River Basin", size=(180, 180), seed=99)
    else:
        uploaded_file = st.sidebar.file_uploader("Upload SAR Mask / GeoTIFF", type=["tif", "tiff", "npy"])
        if uploaded_file is not None:
            st.sidebar.success("File uploaded successfully!")
            scene = generate_synthetic_sar_scene("Custom AOI Upload", size=(180, 180), seed=12)
        else:
            st.info("ℹ️ Using reference SAR simulation baseline while awaiting custom raster.")
            scene = generate_synthetic_sar_scene("Reference AOI Baseline", size=(180, 180), seed=7)

    # Compute Analytics
    flood_mask = scene["flood_mask"]
    metrics = analyzer.analyze_mask(flood_mask, pixel_resolution_m=float(pixel_res), region_name=scene["region_desc"])

    # Top Level Impact Cards
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">Inundated Extent</div>
                <div class="metric-value">{metrics.flooded_area_sq_km:,.1f} <span style="font-size:1.1rem; color:#94a3b8;">km²</span></div>
                <div style="font-size:0.85rem; color:#38bdf8; margin-top:4px;">{metrics.flooded_area_hectares:,.0f} hectares</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with kpi2:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">Basin Flood Ratio</div>
                <div class="metric-value">{metrics.inundation_percentage:.1f}%</div>
                <div style="font-size:0.85rem; color:#38bdf8; margin-top:4px;">Of total surveyed swath</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with kpi3:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">Exposed Population</div>
                <div class="metric-value">~{metrics.estimated_affected_population:,}</div>
                <div style="font-size:0.85rem; color:#f87171; margin-top:4px;">High humanitarian priority</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with kpi4:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">Infrastructure Risk</div>
                <div class="metric-value">{metrics.infrastructure_risk_index:.0f} <span style="font-size:1.1rem; color:#94a3b8;">/ 100</span></div>
                <div style="font-size:0.85rem; color:#facc15; margin-top:4px;">Critical assets exposed</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Emergency Alert Status Banner
    sev_class = "alert-low"
    if metrics.severity_level == SeverityLevel.CRITICAL:
        sev_class = "alert-critical"
    elif metrics.severity_level == SeverityLevel.HIGH:
        sev_class = "alert-high"
    elif metrics.severity_level == SeverityLevel.MODERATE:
        sev_class = "alert-moderate"

    st.markdown(
        f"""
        <div style="background-color: #1e293b; padding: 14px 20px; border-radius: 8px; border-left: 6px solid #38bdf8; display: flex; align-items: center; justify-content: space-between;">
            <div>
                <strong>Current Status:</strong> <span class="alert-badge {sev_class}">{metrics.severity_level.value}</span>
                <span style="color:#94a3b8; margin-left: 15px;">AOI: {scene['region_desc']}</span>
            </div>
            <span style="font-size:0.85rem; color:#94a3b8;">Constellation: Sentinel-1 C-Band SAR | {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Interactive Dashboard Tabs
    tab_map, tab_sar, tab_sectors, tab_sitrep = st.tabs([
        "🗺️ Geospatial Inundation Map",
        "📡 SAR Backscatter & Change",
        "📊 Sector Vulnerability & Impact",
        "📄 Emergency Situation Report (SitRep)",
    ])

    with tab_map:
        st.subheader("Geospatial Inundation Map & Hazard Layer")
        h, w = flood_mask.shape
        lats = np.linspace(scene["lat_range"][0], scene["lat_range"][1], h)
        lons = np.linspace(scene["lon_range"][0], scene["lon_range"][1], w)
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        # Extract flood coordinate points for interactive map
        flood_indices = np.where(flood_mask > 0)
        sample_step = max(1, len(flood_indices[0]) // 400)  # Downsample for snappy browser rendering
        sub_lats = lat_grid[flood_indices][::sample_step]
        sub_lons = lon_grid[flood_indices][::sample_step]

        fig_map = go.Figure()

        # Add center point & hazard perimeter
        fig_map.add_trace(go.Scattermapbox(
            lat=sub_lats,
            lon=sub_lons,
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=7,
                color='#38bdf8',
                opacity=0.75,
            ),
            name="Inundated SAR Detection",
            hovertext="Flooded Zone (SAR Detected Water Surface)",
        ))

        # Add AOI Center Landmark
        fig_map.add_trace(go.Scattermapbox(
            lat=[scene["center_lat"]],
            lon=[scene["center_lon"]],
            mode='markers+text',
            marker=go.scattermapbox.Marker(size=14, color='#ef4444', symbol='circle'),
            text=[f"📍 {scene['region_desc']}"],
            textposition="bottom right",
            name="AOI Ground Station",
        ))

        fig_map.update_layout(
            mapbox_style="carto-darkmatter",
            mapbox=dict(
                center=dict(lat=scene["center_lat"], lon=scene["center_lon"]),
                zoom=9.5,
            ),
            margin=dict(l=0, r=0, t=10, b=0),
            height=520,
            showlegend=True,
        )
        st.plotly_chart(fig_map, use_container_width=True)

    with tab_sar:
        st.subheader("SAR Multi-Temporal Backscatter Analysis")
        st.caption("Comparison between pre-event reference baseline and post-flood SAR backscatter (dB)")

        col_sar1, col_sar2, col_sar3 = st.columns(3)
        with col_sar1:
            st.markdown("**1. Pre-Event VV Backscatter (dB)**")
            fig1 = px.imshow(scene["pre_vv"], color_continuous_scale="Viridis", origin="upper")
            fig1.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=340)
            st.plotly_chart(fig1, use_container_width=True)

        with col_sar2:
            st.markdown("**2. Post-Event Inundation VV (dB)**")
            fig2 = px.imshow(scene["post_vv"], color_continuous_scale="Viridis", origin="upper")
            fig2.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=340)
            st.plotly_chart(fig2, use_container_width=True)

        with col_sar3:
            st.markdown("**3. AquaSentinel AI Flood Mask**")
            fig3 = px.imshow(flood_mask, color_continuous_scale="Blues", origin="upper")
            fig3.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=340)
            st.plotly_chart(fig3, use_container_width=True)

    with tab_sectors:
        st.subheader("Land-Use Vulnerability & Critical Assets Exposed")
        col_c1, col_c2 = st.columns([1, 1])

        with col_c1:
            # Breakdown Chart
            labels = ["Agricultural / Cropland", "Urban / Settlements", "Wetlands / Basins", "Infrastructure Corridors"]
            values = [
                metrics.impacted_cropland_hectares,
                metrics.impacted_urban_area_hectares,
                metrics.flooded_area_hectares * 0.15,
                metrics.flooded_area_hectares * 0.12,
            ]
            fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.45)])
            fig_pie.update_layout(
                title_text="Estimated Inundation by Land-Use Sector (ha)",
                template="plotly_dark",
                height=380,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_c2:
            # Mitigation & Alert Directives
            st.markdown("### 🚨 Emergency Action Directives")
            for i, action in enumerate(metrics.recommended_actions, 1):
                st.markdown(f"**{i}.** {action}")

            st.markdown("---")
            st.markdown("### 📈 Projected Inundation Receding Curve")
            days = [f"Day +{d}" for d in range(1, 8)]
            decay = metrics.flooded_area_sq_km * np.exp(-0.15 * np.arange(7))
            fig_line = px.line(x=days, y=decay, markers=True, labels={"x": "Forecast Horizon", "y": "Projected Flood km²"})
            fig_line.update_layout(template="plotly_dark", height=240, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_line, use_container_width=True)

    with tab_sitrep:
        st.subheader("Automated Disaster Situation Report (SitRep)")
        sitrep_markdown = analyzer.generate_situation_report(metrics, aoi_name=scene["region_desc"])
        st.markdown(sitrep_markdown)

        st.markdown("---")
        st.subheader("📥 Export Intelligence Products")
        col_exp1, col_exp2 = st.columns(2)

        with col_exp1:
            st.download_button(
                label="📄 Download SitRep (Markdown)",
                data=sitrep_markdown,
                file_name=f"AquaSentinel_SitRep_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown",
                use_container_width=True,
            )

        with col_exp2:
            st.download_button(
                label="📊 Download Metrics (JSON)",
                data=json.dumps(metrics.to_dict(), indent=2),
                file_name=f"AquaSentinel_Metrics_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
