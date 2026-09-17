"""
Simplified vector displacement drift simulation module for SAR oil slick tracking.
Provides forward trajectory prediction, backward origin hindcasting, and candidate source replay simulation.
"""

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def simulate_drift(centroid, direction_deg, speed_kmh, hours, reverse=False, km_per_pixel=0.1, num_particles=30):
    """
    Performs a simplified constant-vector particle drift simulation from a starting centroid.

    Note: This is a basic vector displacement model, not a full OpenDrift physics simulation.

    Parameters:
        centroid (tuple): (cx, cy) initial spatial centroid coordinates in pixel space.
        direction_deg (float): Drift heading direction in degrees (0° = North, 90° = East, 180° = South, 270° = West).
        speed_kmh (float): Drift speed in kilometers per hour.
        hours (float): Duration of drift simulation in hours.
        reverse (bool): If True, moves backward in time (opposite vector direction for origin hindcasting).
        km_per_pixel (float): Scale factor converting pixel distance to kilometers (default: 0.1 km/px).
        num_particles (int): Number of scattered slick particles to simulate (default: 30).

    Returns:
        dict: Simulation results containing particle positions, centroids, distance, and probable origin radius.
    """
    cx, cy = float(centroid[0]), float(centroid[1])

    # Generate ~30 scattered initial particles jittered around centroid
    np.random.seed(42)
    jitter_x = np.random.normal(0, 6.0, num_particles)
    jitter_y = np.random.normal(0, 6.0, num_particles)
    initial_pts = np.column_stack((cx + jitter_x, cy + jitter_y))

    # Total distance in km and conversion to pixel displacement
    total_distance_km = float(speed_kmh * hours)
    distance_px = total_distance_km / km_per_pixel if km_per_pixel > 0 else total_distance_km * 10.0

    # Direction angle math (0° North = -y axis, 90° East = +x axis)
    angle_rad = np.radians(90.0 - direction_deg)

    if reverse:
        # Move in exact opposite vector direction for backward hindcasting
        angle_rad += np.pi

    dx = distance_px * np.cos(angle_rad)
    dy = -distance_px * np.sin(angle_rad)

    # Move all particles along constant displacement vector
    drifted_pts = initial_pts + np.array([dx, dy])
    drifted_cx = cx + dx
    drifted_cy = cy + dy

    # Estimated particle dispersion radius for backward origin region
    origin_radius_px = max(18.0, 10.0 + 0.35 * distance_px)

    return {
        "initial_points": initial_pts,
        "drifted_points": drifted_pts,
        "initial_centroid": (cx, cy),
        "drifted_centroid": (drifted_cx, drifted_cy),
        "total_distance_km": round(total_distance_km, 2),
        "direction_deg": direction_deg,
        "hours": hours,
        "reverse": reverse,
        "origin_radius_px": round(origin_radius_px, 1)
    }


def plot_drift_simulation(norm_array, drift_result):
    """
    Generates a matplotlib plot showing the original slick position, vector trajectory arrow,
    and (for backward mode) the shaded probable origin region around the hindcasted centroid.

    Parameters:
        norm_array (np.ndarray): Preprocessed background SAR image array.
        drift_result (dict): Output dictionary from simulate_drift().

    Returns:
        matplotlib.figure.Figure: Generated plot figure.
    """
    fig, ax = plt.subplots(figsize=(7, 7))

    # Render background SAR image
    ax.imshow(norm_array, cmap="gray")

    init_pts = drift_result["initial_points"]
    drift_pts = drift_result["drifted_points"]
    init_c = drift_result["initial_centroid"]
    drift_c = drift_result["drifted_centroid"]
    reverse = drift_result["reverse"]

    # 1. Plot observed initial slick particles and centroid (Orange)
    ax.scatter(init_pts[:, 0], init_pts[:, 1], color="#F97316", s=25, alpha=0.8, label="Observed Slick Position")
    ax.scatter(init_c[0], init_c[1], color="#EA580C", s=90, marker="o", edgecolors="white", linewidth=1.5, zorder=5)

    # 2. Vector displacement trajectory line/arrow
    ax.annotate(
        "",
        xy=(drift_c[0], drift_c[1]),
        xytext=(init_c[0], init_c[1]),
        arrowprops=dict(arrowstyle="->", color="#3B82F6" if not reverse else "#A855F7", lw=2.5, mutation_scale=20)
    )

    if not reverse:
        # Forward mode: Plot future predicted particles (Blue)
        ax.scatter(drift_pts[:, 0], drift_pts[:, 1], color="#2563EB", s=25, alpha=0.8, label="Simulated Forward Drift")
        ax.scatter(drift_c[0], drift_c[1], color="#1D4ED8", s=100, marker="X", label="Predicted Centroid", zorder=5)
        ax.set_title(f"Forward Drift Simulation ({drift_result['total_distance_km']} km in {drift_result['hours']}h)", fontsize=11)
    else:
        # Backward mode: Shaded "Probable origin region" (Circle around hindcasted centroid)
        ax.scatter(drift_pts[:, 0], drift_pts[:, 1], color="#9333EA", s=25, alpha=0.7, label="Hindcast Origin Particles")

        radius = drift_result["origin_radius_px"]
        origin_circle = plt.Circle(
            (drift_c[0], drift_c[1]),
            radius,
            color="#A855F7",
            alpha=0.35,
            label="Probable Origin Region"
        )
        ax.add_patch(origin_circle)
        ax.scatter(drift_c[0], drift_c[1], color="#7E22CE", s=100, marker="*", label="Hindcasted Origin Centroid", zorder=5)
        ax.set_title(f"Backward Origin Hindcasting (-{drift_result['total_distance_km']} km in {drift_result['hours']}h)", fontsize=11)

    ax.set_xlabel("X Pixel Coordinate")
    ax.set_ylabel("Y Pixel Coordinate")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    return fig


def plot_replay_simulation(norm_array, slick_mask, replay_sim, observed_centroid, vessel_name):
    """
    Overlays simulated forward candidate drift particles against actual observed slick mask.

    Parameters:
        norm_array (np.ndarray): Preprocessed background SAR image array.
        slick_mask (np.ndarray): Binary observed oil slick mask (0/255 uint8).
        replay_sim (dict): Output dictionary from simulate_drift() for candidate replay.
        observed_centroid (tuple): (cx, cy) observed slick centroid coordinates.
        vessel_name (str): Name of candidate vessel.

    Returns:
        matplotlib.figure.Figure: Generated overlay plot figure.
    """
    fig, ax = plt.subplots(figsize=(7, 7))

    # Render background SAR image
    ax.imshow(norm_array, cmap="gray")

    # Render observed slick mask in semi-transparent red
    red_mask = np.zeros((*slick_mask.shape, 4), dtype=np.float32)
    red_mask[slick_mask == 255] = [1.0, 0.0, 0.0, 0.45]
    ax.imshow(red_mask)

    # Observed slick centroid (Orange)
    obs_cx, obs_cy = observed_centroid
    ax.scatter(obs_cx, obs_cy, color="#EA580C", s=100, marker="o", edgecolors="white", linewidth=1.5, label="Observed Slick Centroid", zorder=5)

    # Candidate starting position & simulated drift
    drift_pts = replay_sim["drifted_points"]
    vessel_c = replay_sim["initial_centroid"]
    drift_c = replay_sim["drifted_centroid"]

    # Candidate starting position (Purple Vessel marker)
    ax.scatter(vessel_c[0], vessel_c[1], color="#9333EA", s=110, marker="^", edgecolors="white", linewidth=1.5, label=f"Candidate Start: {vessel_name}", zorder=5)

    # Forward drift trajectory arrow
    ax.annotate(
        "",
        xy=(drift_c[0], drift_c[1]),
        xytext=(vessel_c[0], vessel_c[1]),
        arrowprops=dict(arrowstyle="->", color="#9333EA", lw=2.5, mutation_scale=20)
    )

    # Simulated drifted particles & centroid
    ax.scatter(drift_pts[:, 0], drift_pts[:, 1], color="#A855F7", s=25, alpha=0.8, label="Replay Drifted Particles")
    ax.scatter(drift_c[0], drift_c[1], color="#7E22CE", s=110, marker="X", edgecolors="white", linewidth=1.5, label="Replay Simulated Centroid", zorder=5)

    ax.set_title(f"Replay Simulation: {vessel_name}", fontsize=11)
    ax.set_xlabel("X Pixel Coordinate")
    ax.set_ylabel("Y Pixel Coordinate")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    return fig


if __name__ == "__main__":
    start_cx, start_cy = 128.0, 128.0
    fwd = simulate_drift((start_cx, start_cy), direction_deg=45.0, speed_kmh=5.0, hours=12.0, reverse=False)
    bwd = simulate_drift((start_cx, start_cy), direction_deg=45.0, speed_kmh=5.0, hours=12.0, reverse=True)

    print("Forward Drift Centroid:", fwd["drifted_centroid"])
    print("Backward Origin Centroid:", bwd["drifted_centroid"])
