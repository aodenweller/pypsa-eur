# -*- coding: utf-8 -*-
# %%
import logging

import country_converter as coco
import pandas as pd
from _helpers import configure_logging
from gams import transfer as gt

logger = logging.getLogger(__name__)

from types import SimpleNamespace

if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = SimpleNamespace()
        snakemake.input = {
            "region_mapping": "../config/regionmapping_21_EU11.csv",
            "remind_data": "../resources/no_scenario/i1/REMIND2PyPSAEUR.gdx",
        }
        snakemake.config = {
            "countries": ["DE", "FR", "PL"],
            "scenario": {
                "year": [
                    2025,
                    2030,
                    2035,
                    2040,
                    2045,
                    2050,
                    2055,
                    2060,
                    2070,
                    2080,
                    2090,
                    2100,
                    2110,
                    2130,
                    2150,
                ],
                "ll": ["copt"],
                "simpl": [""],
                "clusters": ["4"],
                "opts": ["1H-EpREMIND"],
            },
        }
        snakemake.wildcards = {
            "scenario": "no_scenario",
            "iteration": "1",
        }
        snakemake.output = {
            "co2_price_scenarios": "../resources/no_scenario/i1/co2_price_scenarios.csv"
        }
    else:
        configure_logging(snakemake)

remind_data = gt.Container(snakemake.input["remind_data"])
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

# Limit mapping to countries modelled with PyPSA-EUR
region_mapping = region_mapping.loc[
    region_mapping["PyPSA-EUR"].isin(snakemake.config["countries"])
]
df = remind_data["p_PriceCO2"].records.rename(
    columns={
        "tall": "year",
        "all_regi": "region",
    }
)
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
