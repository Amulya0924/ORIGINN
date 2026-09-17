"""
AIS vessel correlation and candidate filtering module for SAR oil slick investigation.
Filters AIS vessel tracks by time proximity, spatial proximity, transponder gaps, and dynamic threat ranking.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Computes the great-circle distance between two points on Earth in kilometers using Haversine formula.

    Parameters:
        lat1, lon1: Coordinates of first point in decimal degrees.
        lat2, lon2: Coordinates of second point in decimal degrees.

    Returns:
        float: Spatial distance in kilometers.
    """
    R = 6371.0  # Earth's radius in km

    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

    return float(R * c)


def filter_candidates(origin_point, origin_time_str, ais_df, max_distance_km=15.0, max_time_delta_hours=4.0):
    """
    Filters AIS vessel records through a multi-stage investigation pipeline:
    1. Time Proximity Filter
    2. Spatial Proximity Filter
    3. Track-Quality Check (AIS transponder blackout / gap detection)
    4. Candidate Ranking

    Parameters:
        origin_point (tuple): (lat, lon) coordinates of predicted origin centroid.
        origin_time_str (str): Timestamp string of incident / origin time (YYYY-MM-DD HH:MM:SS).
        ais_df (pd.DataFrame): DataFrame containing AIS tracking records.
        max_distance_km (float): Maximum spatial search radius in km (default: 15.0).
        max_time_delta_hours (float): Maximum time window delta in hours (default: 4.0).

    Returns:
        tuple: (ranked_candidates_df, flow_counts)
            - ranked_candidates_df (pd.DataFrame): Ranked list of vessel candidate correlations.
            - flow_counts (dict): Step-by-step pipeline count totals for horizontal flow rendering.
    """
    if ais_df.empty:
        return pd.DataFrame(), {"all_records": 0, "time_filtered": 0, "region_filtered": 0, "track_checked": 0, "candidates": 0}

    df = ais_df.copy()
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"])
    if hasattr(df["timestamp_dt"].dt, 'tz') and df["timestamp_dt"].dt.tz is not None:
        df["timestamp_dt"] = df["timestamp_dt"].dt.tz_localize(None)

    # Parse origin timestamp safely, stripping timezone info to guarantee tz-naive comparison
    try:
        clean_time_str = str(origin_time_str).replace(" UTC", "").replace(" Z", "").strip()
        origin_dt = pd.to_datetime(clean_time_str)
        if hasattr(origin_dt, 'tz') and origin_dt.tz is not None:
            origin_dt = origin_dt.tz_localize(None)
    except Exception:
        origin_dt = pd.to_datetime("2026-09-16 12:30:00")

    # Step 1: All Records Count
    all_records_count = len(df)

    # Step 2: Time Filter
    df["time_delta_hours"] = (df["timestamp_dt"] - origin_dt).abs().dt.total_seconds() / 3600.0
    time_filtered_df = df[df["time_delta_hours"] <= max_time_delta_hours].copy()
    time_filtered_count = len(time_filtered_df)

    if time_filtered_df.empty:
        # Fallback to nearest time if strict filter returns empty
        time_filtered_df = df.copy()

    # Step 3: Region Filter
    orig_lat, orig_lon = origin_point
    time_filtered_df["spatial_dist_km"] = time_filtered_df.apply(
        lambda r: haversine_distance(orig_lat, orig_lon, r["lat"], r["lon"]),
        axis=1
    )
    region_filtered_df = time_filtered_df[time_filtered_df["spatial_dist_km"] <= max_distance_km].copy()
    region_filtered_count = len(region_filtered_df)

    if region_filtered_df.empty:
        region_filtered_df = time_filtered_df.copy()

    # Step 4: Track-Quality Check (AIS transponder gap detection per vessel)
    vessel_gap_flags = {}
    for vessel_name, group in df.sort_values("timestamp_dt").groupby("vessel_name"):
        time_diffs = group["timestamp_dt"].diff().dt.total_seconds() / 3600.0
        max_gap = time_diffs.max() if not time_diffs.isna().all() else 0.0
        has_transponder_gap = bool(max_gap >= 2.0)
        vessel_gap_flags[vessel_name] = {
            "has_gap": has_transponder_gap,
            "max_gap_hours": round(float(max_gap) if not np.isnan(max_gap) else 0.0, 1)
        }

    track_checked_count = len(region_filtered_df)

    # Step 5: Candidate Correlation & Ranking
    vessel_candidates = []
    for vessel_name, group in region_filtered_df.groupby("vessel_name"):
        closest_row = group.loc[group["spatial_dist_km"].idxmin()]

        dist_km = round(float(closest_row["spatial_dist_km"]), 2)
        time_delta_h = round(float(closest_row["time_delta_hours"]), 2)
        gap_info = vessel_gap_flags.get(vessel_name, {"has_gap": False, "max_gap_hours": 0.0})

        # Dynamic Status Computation (High / Medium / Low) based on actual spatial & temporal deltas
        if dist_km <= 2.5 and time_delta_h <= 1.5:
            status = "High"
        elif dist_km <= 6.0 and time_delta_h <= 3.0:
            status = "Medium"
        elif gap_info["has_gap"] and dist_km <= 8.0:
            status = "Medium"
        else:
            status = "Low"

        # Quality flag string
        quality_status = f"Gap Flag ({gap_info['max_gap_hours']}h)" if gap_info["has_gap"] else "Normal Pings"

        vessel_candidates.append({
            "vessel_name": vessel_name,
            "mmsi": closest_row.get("mmsi", "N/A"),
            "spatial_match": f"{dist_km} km",
            "time_match": f"{time_delta_h} h",
            "dist_km_num": dist_km,
            "time_delta_num": time_delta_h,
            "speed_knots": closest_row.get("speed", 0.0),
            "track_quality": quality_status,
            "status": status
        })

    candidates_df = pd.DataFrame(vessel_candidates)

    if not candidates_df.empty:
        # Sort candidates by spatial distance and status rank
        status_rank_map = {"High": 0, "Medium": 1, "Low": 2}
        candidates_df["rank_order"] = candidates_df["status"].map(status_rank_map)
        candidates_df = candidates_df.sort_values(["rank_order", "dist_km_num"]).reset_index(drop=True)
        candidates_df["rank"] = range(1, len(candidates_df) + 1)
        candidates_df = candidates_df.drop(columns=["rank_order", "dist_km_num", "time_delta_num"])
    else:
        candidates_df = pd.DataFrame(columns=["rank", "vessel_name", "mmsi", "time_match", "spatial_match", "speed_knots", "track_quality", "status"])

    flow_counts = {
        "all_records": all_records_count,
        "time_filtered": time_filtered_count,
        "region_filtered": region_filtered_count,
        "track_checked": track_checked_count,
        "candidates": len(candidates_df)
    }

    return candidates_df, flow_counts


if __name__ == "__main__":
    csv_path = Path("data/sample_ais.csv")
    if csv_path.exists():
        ais_data = pd.read_csv(csv_path)
        origin_sample = (55.0125, 13.0125)
        time_sample = "2026-09-16 12:30:00"
        ranked, counts = filter_candidates(origin_sample, time_sample, ais_data)

        print("AIS Filtering Flow Counts:", counts)
        print("\nRanked Vessel Candidates:")
        print(ranked.to_string(index=False))
