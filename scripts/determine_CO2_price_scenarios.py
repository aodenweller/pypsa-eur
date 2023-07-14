# -*- coding: utf-8 -*-
# %%
import logging

import country_converter as coco
import pandas as pd
from _helpers import configure_logging, get_region_mapping, read_remind_data

logger = logging.getLogger(__name__)

from types import SimpleNamespace

if "snakemake" not in globals():
    from types import SimpleNamespace

    # mock_snakemake doesn't work with checkpoints
    snakemake = SimpleNamespace()
    snakemake.wildcards = {
        "scenario": "no_scenario",
        "iteration": "1",
    }

    snakemake.config = {
        "countries": ["DE"],
        "scenario": {
            "simpl": [""],
            "ll": ["copt"],
            "clusters": [4],
            "opts": ["1H-RCL-EpREMIND"],
            "year": [2020, 2025, 2030, 2035, 2040, 2045, 2050],
        },
    }

    snakemake.input = {
        "region_mapping": "../config/regionmapping_21_EU11.csv",
        "remind_data": "../resources/no_scenario/i2/REMIND2PyPSAEUR.gdx",
    }

    snakemake.output = {
        "co2_price_scenarios": "../resources/no_scenario/i1/co2_price_scenarios.csv",
    }
else:
    configure_logging(snakemake)

# Load and transform region mapping
region_mapping = get_region_mapping(
    snakemake.input["region_mapping"], source="PyPSA-EUR", target="REMIND-EU"
)
region_mapping = pd.DataFrame(region_mapping).T.reset_index()
region_mapping.columns = ["PyPSA-EUR", "REMIND-EU"]
region_mapping = region_mapping.loc[
    region_mapping["PyPSA-EUR"].isin(snakemake.config["countries"])
]
# %%
df = read_remind_data(
    file_path=snakemake.input["remind_data"],
    variable_name="p_PriceCO2",
    rename_columns={
        "tall": "year",
        "all_regi": "region",
    },
)

# %%
# Calculate mean co2 price across all regions overlapping between REMIND and PyPSA-EUR countries for each year
df = (
    df.loc[df["region"].isin(region_mapping["REMIND-EU"])]
    .groupby("year")["value"]
    .mean()
)

# add all years from variable as additional indices to df
df = (
    df.reindex(snakemake.config["scenario"]["year"], fill_value=0)
    .to_frame("co2_price")
    .reset_index()
)

# Create a csv file which can be directly read by snakemake paramspace
# need to add the remaining wildcards of interest, assume each wildcard has only one
# entry in the config.yaml file
if (
    len(snakemake.config["scenario"]["ll"]) != 1
    or len(snakemake.config["scenario"]["simpl"]) != 1
    or len(snakemake.config["scenario"]["clusters"]) != 1
    or len(snakemake.config["scenario"]["opts"]) != 1
):
    logger.error(
        "Only exactly one entry for config['scenario'] -> simpl, clusters, ll and opts in config permitted."
    )
# %%
df["scenario"] = snakemake.wildcards["scenario"]
df["iteration"] = snakemake.wildcards["iteration"]
df["simpl"] = snakemake.config["scenario"]["simpl"][0]
df["clusters"] = snakemake.config["scenario"]["clusters"][0]
df["ll"] = snakemake.config["scenario"]["ll"][0]
df["opts"] = snakemake.config["scenario"]["opts"][0]

# Preserve all opts and substitute the co2 price placeholder ("EpREMIND") with the actual co2 price from REMIND
if not df["opts"].str.contains("-EpREMIND").all():
    logging.error("Placeholder '-EpREMIND' missing from config['scenario']['opts']")
df["opts"] = df.apply(
    lambda row: row["opts"].replace("-EpREMIND", f"-Ep{row['co2_price']:0.1f}"),
    axis="columns",
)

# no longer needed
df = df.drop(columns=["co2_price"])

df.to_csv(snakemake.output["co2_price_scenarios"], index=False)
