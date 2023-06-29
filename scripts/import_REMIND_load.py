# -*- coding: utf-8 -*-
import logging
from types import SimpleNamespace

import country_converter as coco
import pandas as pd
from _helpers import configure_logging
from gams import transfer as gt

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = SimpleNamespace()

        snakemake.input = {
            "load_timeseries": "../resources/load.csv",
            "region_mapping": "../config/regionmapping_21_EU11.csv",
            "remind_data": "../resources/no_scenario/coupling-parameters/i1/REMIND2PyPSA.gdx",
        }

        snakemake.output = {
            "load_timeseries": "../resources/no_scenario/i1/load_2020.csv",
        }

        snakemake.wildcards = {
            "year": "2020",
        }
    configure_logging(snakemake)

# Load original load timeseries from PyPSA-EUR
load = pd.read_csv(snakemake.input["load_timeseries"], index_col=0)

# Load REMIND-EU demand data
remind_data = gt.Container(snakemake.input["remind_data"])
demand = remind_data["v32_usableSeDisp"].records
demand = demand.rename(
    columns={"all_enty": "sector", "ttot": "year", "all_regi": "region", "level": "value"}
)
demand = demand.loc[
    (demand["sector"] == "seel") & (demand["year"] == snakemake.wildcards["year"])
]
demand["value"] *= 1e6 * 8760  # Convert from TWa to MWh
demand = demand.set_index(["region"])["value"]

# Create region mapping by loading the original mapping from REMIND-EU from file
# and then mapping ISO 3166-1 alpha-3 country codes to PyPSA-EUR ISO 3166-1 alpha-2 country codes
region_mapping = pd.read_csv(snakemake.input["region_mapping"], sep=";").rename(
    columns={"RegionCode": "REMIND-EU"}
)

region_mapping["PyPSA-EUR"] = coco.convert(
    names=region_mapping["CountryCode"], to="ISO2"
)
region_mapping = region_mapping[["PyPSA-EUR", "REMIND-EU"]]

# Append Kosovo to region mapping, not present in standard mapping and uses non-standard "KV" in PyPSA-EUR
region_mapping = pd.concat(
    [
        region_mapping,
        pd.DataFrame(
            {
                "REMIND-EU": "NES",
                "PyPSA-EUR": "KV",
            },
            columns=["PyPSA-EUR", "REMIND-EU"],
            index=[0],
        ),
    ]
).reset_index(drop=True)

# Regions which are in PyPSA-EUR
region_mapping = region_mapping.loc[region_mapping["PyPSA-EUR"].isin(load.columns)]

# Make sure all regions are mapped
missing_mapping = set(load.columns) - set(region_mapping["PyPSA-EUR"])
if missing_mapping:
    logger.warning(
        f"Missing mapping for the following regions: {missing_mapping}. These regions will be ignored."
    )

# Calculate load from PyPSA-EUR load-timeseries for REMIND-EU regions
regional_load = (
    region_mapping.join(load.sum(axis="rows").rename("annual_load"), on="PyPSA-EUR")
    .groupby("REMIND-EU")["annual_load"]
    .sum()
)

# Factor by which PyPSA-EUR loads have to be scaled to match REMIND-EU demand
load_scaling_factor = (demand / regional_load).dropna().rename("load_scaling_factor")
load_scaling_factor = region_mapping.join(
    load_scaling_factor, on="REMIND-EU"
).set_index("PyPSA-EUR")["load_scaling_factor"]

# Scale load timeseries
scaled_load = load * load_scaling_factor

# Check whether differences after scaling is negligible
if not all(
    (
        demand
        - region_mapping.join(
            scaled_load.sum(axis="rows").rename("annual_load"), on="PyPSA-EUR"
        )
        .groupby("REMIND-EU")["annual_load"]
        .sum()
    ).dropna()
    < 1e-5
):
    logger.warning("Scaled load values do not seem to match REMIND-EU demand values")

scaled_load.to_csv(snakemake.output["load_timeseries"])
