#!/usr/bin/env python3
"""Create 2015–2017 to 2021–2023 environmental and modelled change outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import TwoSlopeNorm
from rasterio.enums import Resampling


PREDICTION_NODATA = -9999.0
RELIABILITY_NODATA = 255
SPECIES = {
    1: "General_Conifer",
    2: "General_Broadleaf",
    3: "Acer",
    4: "Alnus",
    5: "Betula",
    6: "Pinus",
    7: "Prunus",
    8: "Quercus",
    9: "Sorbus",
    10: "Tilia",
    11: "Ulmus",
}
ENVIRONMENT_CHANGES = {
    "Tree_density_25m": ("Density_15", "Density_21", "trees within 25 m"),
    "Monoculture_rate": ("Mono_Rate_", "Mono_Rat_1", "proportion"),
    "Sky_view_factor": ("svf15_17", "svf21_23", "proportion"),
    "Solar_radiation": ("RA15_17", "RA21_23", "source units"),
    "Land_surface_temperature": ("LST15_17", "LST21_23", "°C"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--spatial-output", required=True, type=Path)
    parser.add_argument("--chunk-size", type=int, default=250_000)
    return parser.parse_args()


def distribution_record(
    variable: str,
    values: np.ndarray,
    scope: str,
    unit: str,
) -> dict:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "Variable": variable,
            "Scope": scope,
            "Unit": unit,
            "N": 0,
        }
    return {
        "Variable": variable,
        "Scope": scope,
        "Unit": unit,
        "N": int(finite.size),
        "Min": float(np.min(finite)),
        "P05": float(np.quantile(finite, 0.05)),
        "Median": float(np.median(finite)),
        "Mean": float(np.mean(finite)),
        "P95": float(np.quantile(finite, 0.95)),
        "Max": float(np.max(finite)),
        "Percent_Positive": float(100.0 * np.mean(finite > 0)),
        "Percent_Negative": float(100.0 * np.mean(finite < 0)),
        "Percent_Zero": float(100.0 * np.mean(finite == 0)),
    }


def create_environment_change_raster(
    args: argparse.Namespace,
    grid: dict,
    profile: dict,
) -> tuple[Path, pd.DataFrame]:
    output_path = args.spatial_output / "environment_change_2015_2023.tif"
    height = int(grid["height"])
    width = int(grid["width"])
    cell_count = height * width
    arrays = {
        name: np.full(cell_count, PREDICTION_NODATA, dtype=np.float32)
        for name in ENVIRONMENT_CHANGES
    }
    columns = ["X", "Y"]
    for start, end, _ in ENVIRONMENT_CHANGES.values():
        columns.extend([start, end])
    columns = list(dict.fromkeys(columns))

    min_x = float(grid["min_x_center"])
    max_y = float(grid["max_y_center"])
    resolution = float(grid["resolution"])
    for chunk_number, chunk in enumerate(
        pd.read_csv(args.input_csv, usecols=columns, chunksize=args.chunk_size),
        start=1,
    ):
        x = pd.to_numeric(chunk["X"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(chunk["Y"], errors="coerce").to_numpy(dtype=float)
        col = np.rint((x - min_x) / resolution).astype(np.int64)
        row = np.rint((max_y - y) / resolution).astype(np.int64)
        valid_xy = (
            np.isfinite(x)
            & np.isfinite(y)
            & (row >= 0)
            & (row < height)
            & (col >= 0)
            & (col < width)
        )
        flat = row[valid_xy] * width + col[valid_xy]
        for name, (start, end, _) in ENVIRONMENT_CHANGES.items():
            start_value = pd.to_numeric(
                chunk[start], errors="coerce"
            ).to_numpy(dtype=float)
            end_value = pd.to_numeric(chunk[end], errors="coerce").to_numpy(
                dtype=float
            )
            valid_value = (
                valid_xy
                & np.isfinite(start_value)
                & np.isfinite(end_value)
                & (start_value > -9990.0)
                & (end_value > -9990.0)
            )
            local_flat = row[valid_value] * width + col[valid_value]
            arrays[name][local_flat] = (
                end_value[valid_value] - start_value[valid_value]
            ).astype(np.float32)
        if chunk_number % 5 == 0:
            print(
                f"[environment] processed {chunk_number * args.chunk_size:,} rows",
                flush=True,
            )

    change_profile = {
        **profile,
        "count": len(ENVIRONMENT_CHANGES),
        "dtype": "float32",
        "nodata": PREDICTION_NODATA,
        "compress": "DEFLATE",
        "predictor": 3,
        "zlevel": 6,
        "BIGTIFF": "IF_SAFER",
    }
    summary = []
    with rasterio.open(output_path, "w", **change_profile) as destination:
        for band, (name, (_, _, unit)) in enumerate(
            ENVIRONMENT_CHANGES.items(), start=1
        ):
            raster = arrays[name].reshape(height, width)
            destination.write(raster, band)
            destination.set_band_description(band, name)
            destination.update_tags(
                band,
                change_definition="2021-2023 value minus 2015-2017 value",
                unit=unit,
            )
            values = raster[raster != PREDICTION_NODATA].astype(float)
            summary.append(
                distribution_record(name, values, "all_valid_grid_cells", unit)
            )
    return output_path, pd.DataFrame(summary)


def create_modelled_change_rasters(
    output: Path,
) -> tuple[Path, Path, Path, pd.DataFrame]:
    early_pct = rasterio.open(output / "relative_growth_pct_15_17.tif")
    late_pct = rasterio.open(output / "relative_growth_pct_21_23.tif")
    early_level = rasterio.open(output / "suitability_level_15_17.tif")
    late_level = rasterio.open(output / "suitability_level_21_23.tif")
    early_reliability = rasterio.open(output / "reliability_15_17.tif")
    late_reliability = rasterio.open(output / "reliability_21_23.tif")

    pct_path = output / "relative_growth_change_pp_2015_2023.tif"
    level_path = output / "suitability_level_change_2015_2023.tif"
    reliability_path = output / "reliability_intersection_2015_2023.tif"
    pct_profile = {
        **early_pct.profile,
        "count": len(SPECIES),
        "dtype": "float32",
        "nodata": PREDICTION_NODATA,
        "compress": "DEFLATE",
        "predictor": 3,
        "zlevel": 6,
        "BIGTIFF": "IF_SAFER",
    }
    level_profile = {
        **early_level.profile,
        "count": len(SPECIES),
        "dtype": "int8",
        "nodata": -128,
        "compress": "DEFLATE",
        "predictor": 2,
        "zlevel": 6,
        "BIGTIFF": "IF_SAFER",
    }
    reliability_profile = {
        **early_reliability.profile,
        "count": 1,
        "dtype": "uint8",
        "nodata": RELIABILITY_NODATA,
        "compress": "DEFLATE",
        "predictor": 2,
        "zlevel": 6,
    }
    early_reliable = early_reliability.read(1)
    late_reliable = late_reliability.read(1)
    populated = (
        (early_reliable != RELIABILITY_NODATA)
        & (late_reliable != RELIABILITY_NODATA)
    )
    common_reliable = populated & (early_reliable == 1) & (late_reliable == 1)
    reliability = np.full(early_reliable.shape, RELIABILITY_NODATA, dtype=np.uint8)
    reliability[populated] = common_reliable[populated].astype(np.uint8)

    summaries = []
    with (
        rasterio.open(pct_path, "w", **pct_profile) as pct_destination,
        rasterio.open(level_path, "w", **level_profile) as level_destination,
        rasterio.open(
            reliability_path, "w", **reliability_profile
        ) as reliability_destination,
    ):
        reliability_destination.write(reliability, 1)
        reliability_destination.set_band_description(
            1, "Reliable_in_both_15_17_and_21_23"
        )
        reliability_destination.update_tags(
            1,
            value_1="inside development min-max domain in both periods",
            value_0="outside domain in one or both periods",
        )
        for band, species_name in SPECIES.items():
            early = early_pct.read(band)
            late = late_pct.read(band)
            valid = (
                (early != early_pct.nodata)
                & (late != late_pct.nodata)
                & np.isfinite(early)
                & np.isfinite(late)
            )
            pct_change = np.full(early.shape, PREDICTION_NODATA, dtype=np.float32)
            pct_change[valid] = late[valid] - early[valid]
            pct_destination.write(pct_change, band)
            pct_destination.set_band_description(band, f"Sp{band}_{species_name}")
            pct_destination.update_tags(
                band,
                change_definition=(
                    "2021-2023 modelled annual growth percent minus "
                    "2015-2017 modelled annual growth percent"
                ),
                unit="percentage points per year",
            )

            early_class = early_level.read(band).astype(np.int16)
            late_class = late_level.read(band).astype(np.int16)
            class_valid = (early_class > 0) & (late_class > 0)
            class_change = np.full(early.shape, -128, dtype=np.int8)
            class_change[class_valid] = (
                late_class[class_valid] - early_class[class_valid]
            ).astype(np.int8)
            level_destination.write(class_change, band)
            level_destination.set_band_description(
                band, f"Sp{band}_{species_name}"
            )
            level_destination.update_tags(
                band,
                change_definition="2021-2023 suitability level minus 2015-2017 level",
                unit="level",
            )

            all_values = pct_change[valid].astype(float)
            reliable_values = pct_change[valid & common_reliable].astype(float)
            summaries.append(
                {
                    "Species_Code": band,
                    "Species_Name": species_name,
                    **distribution_record(
                        "Modelled_relative_growth_change",
                        all_values,
                        "all_modelled_grid_cells",
                        "percentage points per year",
                    ),
                }
            )
            summaries.append(
                {
                    "Species_Code": band,
                    "Species_Name": species_name,
                    **distribution_record(
                        "Modelled_relative_growth_change",
                        reliable_values,
                        "strictly_reliable_in_both_periods",
                        "percentage points per year",
                    ),
                }
            )

    for dataset in [
        early_pct,
        late_pct,
        early_level,
        late_level,
        early_reliability,
        late_reliability,
    ]:
        dataset.close()
    return pct_path, level_path, reliability_path, pd.DataFrame(summaries)


def read_display(path: Path, band: int, maximum_pixels: int = 800):
    with rasterio.open(path) as source:
        scale = max(source.height / maximum_pixels, source.width / maximum_pixels, 1)
        height = max(1, round(source.height / scale))
        width = max(1, round(source.width / scale))
        data = source.read(
            band,
            out_shape=(height, width),
            resampling=Resampling.nearest,
        ).astype(float)
        nodata = source.nodata
    if nodata is not None:
        data[data == nodata] = np.nan
    return data


def robust_symmetric_limit(data: np.ndarray, quantile: float = 0.99) -> float:
    finite = np.abs(data[np.isfinite(data)])
    if finite.size == 0:
        return 1.0
    limit = float(np.quantile(finite, quantile))
    return max(limit, np.finfo(float).eps)


def create_figure7(
    output: Path,
    environment_path: Path,
    model_change_path: Path,
    reliability_path: Path,
) -> Path:
    figure_path = output / "Figure7_environment_change_2015_2023.png"
    panels = [
        (environment_path, 1, "A  Tree density", "Δ trees within 25 m"),
        (environment_path, 2, "B  Monoculture rate", "Δ proportion"),
        (environment_path, 3, "C  Sky-view factor", "Δ proportion"),
        (environment_path, 4, "D  Solar radiation", "Δ source units"),
        (environment_path, 5, "E  Land-surface temperature", "Δ °C"),
        (
            model_change_path,
            2,
            "F  General broadleaf modelled growth",
            "Δ annual percentage points",
        ),
    ]
    common_reliable = read_display(reliability_path, 1)
    figure, axes = plt.subplots(2, 3, figsize=(15.6, 10.0))
    for index, (path, band, title, label) in enumerate(panels):
        axis = axes.flat[index]
        data = read_display(path, band)
        if index == 5:
            data[common_reliable != 1] = np.nan
        limit = robust_symmetric_limit(data)
        image = axis.imshow(
            data,
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
            interpolation="nearest",
        )
        axis.set_title(title, fontsize=11.5, weight="bold", loc="left")
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
        colorbar = figure.colorbar(
            image, ax=axis, orientation="horizontal", fraction=0.045, pad=0.025
        )
        colorbar.set_label(label, fontsize=9)
        colorbar.ax.tick_params(labelsize=8)
    figure.suptitle(
        "Environmental and modelled relative-growth change: "
        "2015–2017 to 2021–2023",
        fontsize=15,
        weight="bold",
        y=0.995,
    )
    figure.text(
        0.5,
        0.007,
        (
            "Change = later minus earlier. Colour limits show the symmetric 99th "
            "percentile for each panel. Grey/blank in F is outside the strict "
            "training domain in at least one period."
        ),
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=[0, 0.025, 1, 0.975])
    figure.savefig(
        figure_path, dpi=300, bbox_inches="tight", facecolor="#EFEFEF"
    )
    plt.close(figure)
    return figure_path


def main() -> None:
    args = parse_args()
    grid = json.loads(
        (args.spatial_output / "grid_metadata.json").read_text(encoding="utf-8")
    )
    with rasterio.open(
        args.spatial_output / "relative_growth_pct_15_17.tif"
    ) as template:
        profile = template.profile

    environment_path, environment_summary = create_environment_change_raster(
        args, grid, profile
    )
    (
        model_change_path,
        level_change_path,
        reliability_path,
        model_summary,
    ) = create_modelled_change_rasters(args.spatial_output)
    summary = pd.concat(
        [environment_summary, model_summary], ignore_index=True, sort=False
    )
    summary.to_csv(
        args.spatial_output / "change_detection_summary.csv", index=False
    )
    figure_path = create_figure7(
        args.spatial_output,
        environment_path,
        model_change_path,
        reliability_path,
    )

    (args.spatial_output / "CHANGE_DETECTION_GUIDE.md").write_text(
        "\n".join(
            [
                "# 2015–2023 change-detection outputs",
                "",
                "The available layers represent 2015–2017 and 2021–2023 periods; "
                "the reported change is later minus earlier and should not be "
                "interpreted as two exact single-year observations.",
                "",
                f"- `{environment_path.name}`: five measured/mapped environmental changes.",
                f"- `{model_change_path.name}`: change in modelled annual relative growth, 11 species bands.",
                f"- `{level_change_path.name}`: change in the fixed suitability level, 11 species bands.",
                f"- `{reliability_path.name}`: strict-domain intersection mask.",
                f"- `{figure_path.name}`: manuscript-ready Figure 7 candidate.",
                "- Noise and night illumination were not included in temporal change because the input provides only one shared layer for all periods.",
                "- Modelled growth change isolates the effect of period-varying mapped inputs while species, 10 m height, park context, model, and classification thresholds remain fixed.",
                "- The result is model-based change detection and is not evidence of a causal environmental effect.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Created {figure_path}", flush=True)


if __name__ == "__main__":
    main()
