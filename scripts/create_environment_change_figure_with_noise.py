#!/usr/bin/env python3
"""Create the six-panel environmental-change figure with temporal noise change.

Panels A-E use the existing later-minus-earlier environmental-change raster.
Panel F reconstructs combined daytime noise for 2017 and 2022 from the four
available municipal transport-noise sources and maps 2022 minus 2017.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.warp import reproject


NODATA = -9999.0
QUIET_FLOOR_DB = 40.0
NOISE_MODES = ("road", "rail", "tram", "metro")
SOURCE_YEARS = ("17", "22")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment-change", required=True, type=Path)
    parser.add_argument("--noise-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def robust_symmetric_limit(data: np.ndarray, quantile: float = 0.99) -> float:
    finite = np.abs(data[np.isfinite(data)])
    if finite.size == 0:
        return 1.0
    return max(float(np.quantile(finite, quantile)), np.finfo(float).eps)


def distribution(values: np.ndarray, variable: str, unit: str) -> dict:
    finite = values[np.isfinite(values)]
    return {
        "Variable": variable,
        "Unit": unit,
        "N": int(finite.size),
        "Min": float(np.min(finite)),
        "P01": float(np.quantile(finite, 0.01)),
        "P05": float(np.quantile(finite, 0.05)),
        "Median": float(np.median(finite)),
        "Mean": float(np.mean(finite)),
        "P95": float(np.quantile(finite, 0.95)),
        "P99": float(np.quantile(finite, 0.99)),
        "Max": float(np.max(finite)),
        "Percent_negative": float(100.0 * np.mean(finite < 0)),
        "Percent_zero": float(100.0 * np.mean(finite == 0)),
        "Percent_positive": float(100.0 * np.mean(finite > 0)),
    }


def reconstruct_noise_year(
    year: str,
    noise_root: Path,
    shape: tuple[int, int],
    transform,
    domain_valid: np.ndarray,
) -> np.ndarray:
    """Return the maximum transport-source daytime class midpoint in dB."""
    combined = np.full(shape, QUIET_FLOOR_DB, dtype=np.float32)
    source_found = np.zeros(shape, dtype=bool)
    for mode in NOISE_MODES:
        source_path = noise_root / f"_{year}{mode}d"
        with rasterio.open(source_path) as source:
            destination = np.full(shape, np.nan, dtype=np.float32)
            # The ArcInfo grids carry an unnamed but parameter-equivalent GK25FIN
            # definition. EPSG:3879 is stated explicitly to avoid WKT ambiguity.
            reproject(
                source=rasterio.band(source, 1),
                destination=destination,
                src_transform=source.transform,
                src_crs=CRS.from_epsg(3879),
                src_nodata=source.nodata,
                dst_transform=transform,
                dst_crs=CRS.from_epsg(3879),
                dst_nodata=np.nan,
                resampling=Resampling.nearest,
            )
        valid = np.isfinite(destination) & (destination >= 45) & (destination <= 80)
        # Source grids encode the lower edge of 5-dB classes (45, 50, ...).
        # The tree table used the class midpoints (47.5, 52.5, ...).
        midpoint = destination + 2.5
        combined[valid] = np.maximum(combined[valid], midpoint[valid])
        source_found |= valid

    combined[~domain_valid] = NODATA
    return combined


def write_noise_rasters(
    output: Path,
    profile: dict,
    early: np.ndarray,
    late: np.ndarray,
    change: np.ndarray,
) -> tuple[Path, Path, Path]:
    common_profile = {
        **profile,
        "count": 1,
        "dtype": "float32",
        "nodata": NODATA,
        "compress": "DEFLATE",
        "predictor": 3,
        "zlevel": 6,
    }
    products = (
        (
            output / "daytime_noise_2017_db.tif",
            early,
            "Combined_daytime_noise_2017_dB",
            "Maximum of road, rail, tram and metro daytime-noise class midpoints; 40 dB quiet floor",
        ),
        (
            output / "daytime_noise_2022_db.tif",
            late,
            "Combined_daytime_noise_2022_dB",
            "Maximum of road, rail, tram and metro daytime-noise class midpoints; 40 dB quiet floor",
        ),
        (
            output / "daytime_noise_change_2017_2022_db.tif",
            change,
            "Daytime_noise_change_2022_minus_2017_dB",
            "Later minus earlier: combined 2022 daytime noise minus combined 2017 daytime noise",
        ),
    )
    paths = []
    for path, data, description, note in products:
        with rasterio.open(path, "w", **common_profile) as destination:
            destination.write(data.astype(np.float32), 1)
            destination.set_band_description(1, description)
            destination.update_tags(
                1,
                unit="dB",
                note=note,
                quiet_floor_db=QUIET_FLOOR_DB,
            )
        paths.append(path)
    return tuple(paths)


def downsample(data: np.ndarray, maximum_pixels: int = 900) -> np.ndarray:
    step = max(1, int(np.ceil(max(data.shape) / maximum_pixels)))
    return data[::step, ::step]


def create_figure(
    output: Path,
    environment_path: Path,
    noise_change: np.ndarray,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    with rasterio.open(environment_path) as source:
        environment = [source.read(i).astype(float) for i in range(1, 6)]
        nodata = source.nodata
    if nodata is not None:
        for data in environment:
            data[data == nodata] = np.nan
    noise = noise_change.astype(float)
    noise[noise == NODATA] = np.nan

    panels = [
        (environment[0], "A  Tree density", "Δ trees within 25 m", False),
        (environment[1], "B  Monoculture rate", "Δ proportion", True),
        (environment[2], "C  Sky-view factor", "Δ proportion", False),
        (environment[3], "D  Solar radiation", "Δ source units", True),
        (environment[4], "E  Land-surface temperature", "Δ °C", True),
        (noise, "F  Daytime noise", "Δ dB", True),
    ]
    standard_palette = LinearSegmentedColormap.from_list(
        "negative_red_positive_green",
        ["#b2182b", "#ef8a62", "#fffdf5", "#7fbf7b", "#1b7837"],
        N=256,
    )
    reversed_palette = standard_palette.reversed(
        name="negative_green_positive_red"
    )

    figure, axes = plt.subplots(2, 3, figsize=(15.6, 10.2), facecolor="white")
    for axis, (full_data, title, label, reverse_colours) in zip(axes.flat, panels):
        data = downsample(full_data)
        limit = robust_symmetric_limit(data)
        image = axis.imshow(
            data,
            cmap=reversed_palette if reverse_colours else standard_palette,
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
            interpolation="nearest",
        )
        axis.set_title(title, fontsize=11.5, weight="bold", loc="left", pad=6)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
        colorbar = figure.colorbar(
            image,
            ax=axis,
            orientation="horizontal",
            fraction=0.045,
            pad=0.025,
            extend="both",
        )
        colorbar.set_label(label, fontsize=9)
        colorbar.ax.tick_params(labelsize=8)

    figure.suptitle(
        "Mapped environmental change: 2015–2017 to 2021–2023",
        fontsize=15,
        weight="bold",
        y=0.992,
    )
    figure.text(
        0.5,
        0.008,
        (
            "Change = later minus earlier. In A and C, red = decrease and green = increase; "
            "in B, D, E and F, green = negative and red = positive. Pale colours indicate "
            "little or no change. Limits are panel-specific symmetric 99th percentiles."
        ),
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=[0, 0.035, 1, 0.965])

    stem = output / "Figure7_environment_change_with_noise_2015_2023"
    paths = []
    for suffix, kwargs in (
        (".png", {"dpi": 600}),
        (".pdf", {}),
        (".svg", {}),
    ):
        path = stem.with_suffix(suffix)
        figure.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        paths.append(path)
    plt.close(figure)
    return paths


def save_figure_data(
    output: Path,
    environment_path: Path,
    noise_change: np.ndarray,
) -> Path:
    with rasterio.open(environment_path) as source:
        arrays = [source.read(i).astype(float) for i in range(1, 6)]
        nodata = source.nodata
    if nodata is not None:
        for data in arrays:
            data[data == nodata] = np.nan
    noise = noise_change.astype(float)
    noise[noise == NODATA] = np.nan
    figure_data = output / "Figure7_environment_change_with_noise_data.npz"
    np.savez_compressed(
        figure_data,
        panel_a=downsample(arrays[0]),
        panel_b=downsample(arrays[1]),
        panel_c=downsample(arrays[2]),
        panel_d=downsample(arrays[3]),
        panel_e=downsample(arrays[4]),
        panel_f=downsample(noise),
    )
    return figure_data


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with rasterio.open(args.environment_change) as environment:
        profile = environment.profile
        first = environment.read(1)
        domain_valid = np.isfinite(first)
        if environment.nodata is not None:
            domain_valid &= first != environment.nodata
        shape = (environment.height, environment.width)
        transform = environment.transform

    early = reconstruct_noise_year(
        SOURCE_YEARS[0], args.noise_root, shape, transform, domain_valid
    )
    late = reconstruct_noise_year(
        SOURCE_YEARS[1], args.noise_root, shape, transform, domain_valid
    )
    change = np.full(shape, NODATA, dtype=np.float32)
    change[domain_valid] = late[domain_valid] - early[domain_valid]
    raster_paths = write_noise_rasters(args.output, profile, early, late, change)
    figure_data_path = save_figure_data(
        args.output, args.environment_change, change
    )
    figure_paths = []
    if not args.prepare_only:
        figure_paths = create_figure(
            args.output, args.environment_change, change
        )

    valid_change = change[change != NODATA].astype(float)
    summary = distribution(valid_change, "Daytime_noise_change_2022_minus_2017", "dB")
    pd.DataFrame([summary]).to_csv(
        args.output / "daytime_noise_change_summary.csv", index=False
    )
    metadata = {
        "change_definition": "later minus earlier",
        "environmental_periods": "2015-2017 to 2021-2023",
        "noise_source_years": "2017 to 2022",
        "noise_sources": list(NOISE_MODES),
        "noise_combination": "maximum available 5-dB class midpoint across transport sources",
        "quiet_floor_db": QUIET_FLOOR_DB,
        "colour_direction": {
            "panels_A_C": "red=negative/decrease; green=positive/increase",
            "panels_B_D_E_F": "green=negative/decrease; red=positive/increase",
        },
        "colour_limits": "panel-specific symmetric 99th percentile",
        "raster_outputs": [str(path) for path in raster_paths],
        "figure_outputs": [str(path) for path in figure_paths],
        "figure_data": str(figure_data_path),
        "noise_change_summary": summary,
    }
    (args.output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
