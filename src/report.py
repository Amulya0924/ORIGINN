"""
Report generation module for ORIGIN SAR Oil Slick Investigation Workspace.
Compiles scene metadata, slick fingerprinting, drift simulation, vessel correlation,
and evidence summary into structured JSON and plain-text investigation packages.
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def generate_report(scene_id, fingerprint, drift_params, top_candidate, evidence_summary):
    """
    Compiles session investigation findings into JSON and formatted plain-text reports.

    Parameters:
        scene_id (str): Name or ID of the SAR scene analyzed.
        fingerprint (dict): Geometric fingerprint extracted from slick mask.
        drift_params (dict): Environmental vector settings and simulation output.
        top_candidate (dict): Vessel candidate details and correlation metadata.
        evidence_summary (dict): Dynamic evidence summary findings and qualitative result.

    Returns:
        tuple: (json_dict, formatted_text_report)
            - json_dict (dict): Complete structured JSON data.
            - formatted_text_report (str): Plain text formatted investigation summary.
    """
    now_utc = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

    # Build structured JSON dictionary
    report_dict = {
        "report_metadata": {
            "title": "ORIGIN SAR Oil Slick Incident Investigation Report",
            "generated_timestamp": now_utc,
            "scene_id": str(scene_id)
        },
        "slick_fingerprint": fingerprint if isinstance(fingerprint, dict) else {},
        "drift_simulation": drift_params if isinstance(drift_params, dict) else {},
        "top_vessel_candidate": top_candidate if isinstance(top_candidate, dict) else {},
        "evidence_summary": evidence_summary if isinstance(evidence_summary, dict) else {}
    }

    # Extract clean display variables
    obs_time = fingerprint.get("observation_time", now_utc) if isinstance(fingerprint, dict) else now_utc
    area_px = fingerprint.get("area_pixels", "N/A") if isinstance(fingerprint, dict) else "N/A"
    area_pct = fingerprint.get("area_pct", "N/A") if isinstance(fingerprint, dict) else "N/A"
    shape_prof = f"{fingerprint.get('shape_classification', 'N/A')} ({fingerprint.get('elongation_ratio', 'N/A')}:1)" if isinstance(fingerprint, dict) else "N/A"
    angle_deg = f"{fingerprint.get('angle_deg', 'N/A')}°" if isinstance(fingerprint, dict) else "N/A"
    centroid_str = f"{fingerprint.get('centroid', ('N/A', 'N/A'))}" if isinstance(fingerprint, dict) else "N/A"

    drift_dir = drift_params.get("direction_deg", "N/A") if isinstance(drift_params, dict) else "N/A"
    drift_spd = drift_params.get("speed_kmh", "N/A") if isinstance(drift_params, dict) else "N/A"
    drift_hrs = drift_params.get("hours", "N/A") if isinstance(drift_params, dict) else "N/A"
    drift_dist = drift_params.get("total_distance_km", "N/A") if isinstance(drift_params, dict) else "N/A"

    vessel_name = top_candidate.get("vessel_name", "N/A") if isinstance(top_candidate, dict) else "N/A"
    vessel_mmsi = top_candidate.get("mmsi", "N/A") if isinstance(top_candidate, dict) else "N/A"
    vessel_dist = top_candidate.get("spatial_match", "N/A") if isinstance(top_candidate, dict) else "N/A"
    vessel_time = top_candidate.get("time_match", "N/A") if isinstance(top_candidate, dict) else "N/A"
    vessel_quality = top_candidate.get("track_quality", "N/A") if isinstance(top_candidate, dict) else "N/A"

    in_origin = evidence_summary.get("in_origin_region", "N/A") if isinstance(evidence_summary, dict) else "N/A"
    origin_dist = evidence_summary.get("dist_from_origin_center_km", "N/A") if isinstance(evidence_summary, dict) else "N/A"
    aligned_time = evidence_summary.get("aligned_release_window", "N/A") if isinstance(evidence_summary, dict) else "N/A"
    time_offset = evidence_summary.get("time_offset_hours", "N/A") if isinstance(evidence_summary, dict) else "N/A"
    sim_dist = evidence_summary.get("simulated_trajectory_distance_km", "N/A") if isinstance(evidence_summary, dict) else "N/A"
    track_qual = evidence_summary.get("ais_track_quality", vessel_quality) if isinstance(evidence_summary, dict) else vessel_quality
    result_label = evidence_summary.get("result_consistency_label", "Low consistency") if isinstance(evidence_summary, dict) else "Low consistency"

    # Build formatted plain-text report
    text_lines = [
        "================================================================================",
        "           ORIGIN SAR OIL SLICK INCIDENT INVESTIGATION REPORT",
        "================================================================================",
        f"Generated Timestamp : {now_utc}",
        f"Scene Identifier    : {scene_id}",
        "",
        "1. SLICK GEOMETRIC FINGERPRINT",
        "--------------------------------------------------------------------------------",
        f"Observation Time    : {obs_time}",
        f"Slick Area          : {area_px} pixels ({area_pct}% of scene)",
        f"Shape Profile       : {shape_prof}",
        f"Orientation Angle   : {angle_deg}",
        f"Spatial Centroid    : {centroid_str}",
        "Risk Note           : Look-alike risk: requires confirmation",
        "",
        "2. DRIFT RECONSTRUCTION (Simplified Vector Model)",
        "--------------------------------------------------------------------------------",
        f"Drift Vector Heading: {drift_dir}°",
        f"Drift Speed         : {drift_spd} km/h",
        f"Duration            : {drift_hrs} hours",
        f"Displacement Dist   : {drift_dist} km",
        "Model Note          : Constant-vector approximation (OpenDrift physics planned)",
        "",
        "3. CORRELATED AIS VESSEL CANDIDATE",
        "--------------------------------------------------------------------------------",
        f"Vessel Name         : {vessel_name}",
        f"MMSI                : {vessel_mmsi}",
        f"Spatial Proximity   : {vessel_dist}",
        f"Time Proximity      : {vessel_time}",
        f"Track Quality       : {vessel_quality}",
        "",
        "4. DYNAMIC EVIDENCE SUMMARY",
        "--------------------------------------------------------------------------------",
        f"- Present within probable origin region: {in_origin} ({origin_dist} km from region center)",
        f"- Historical position aligned with release window: {aligned_time} ({time_offset} hours offset)",
        f"- Simulated trajectory distance from observed slick: {sim_dist} km",
        f"- AIS track quality: {track_qual}",
        "",
        f"Result: {result_label}",
        "================================================================================"
    ]

    formatted_text = "\n".join(text_lines)

    return report_dict, formatted_text


if __name__ == "__main__":
    # Test report generation
    fp = {"area_pixels": 3888, "area_pct": 5.93, "shape_classification": "Elongated", "elongation_ratio": 2.95, "angle_deg": -59.0, "centroid": (128.0, 128.0)}
    dp = {"direction_deg": 45.0, "speed_kmh": 5.0, "hours": 12.0, "total_distance_km": 60.0}
    tc = {"vessel_name": "Tanker Blue Star", "mmsi": 211890123, "spatial_match": "0.19 km", "time_match": "2.25 h", "track_quality": "Gap Flag (4.2h)"}
    es = {
        "in_origin_region": "Yes",
        "dist_from_origin_center_km": 0.19,
        "aligned_release_window": "Yes",
        "time_offset_hours": 0.5,
        "simulated_trajectory_distance_km": 2.14,
        "ais_track_quality": "Gap Flag (4.2h)",
        "result_consistency_label": "High physical consistency"
    }

    r_dict, r_text = generate_report("palsar_0.png", fp, dp, tc, es)
    print("Test JSON Keys:", list(r_dict.keys()))
    print("\nTest Text Report:\n" + r_text)
