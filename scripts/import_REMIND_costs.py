# -*- coding: utf-8 -*-
# %%

import logging
from types import SimpleNamespace

import country_converter as coco
import numpy as np
import pandas as pd
from _helpers import configure_logging
from gams import transfer as gt

logger = logging.getLogger(__name__)

# TODO mapping hard-coded for now and redundant with extract_coupling_parameters.py
# TODO remove redundancy and outsource into external file
# Only Generation technologies (PyPSA "generator" "carriers")
# Use a two step mapping approach between PyPSA-EUR and REMIND:
# First mapping is aggregating PyPSA-EUR technologies to general technologies
# Second mapping is disaggregating general technologies to REMIND technologies
map_pypsaeur_to_general = {
    "CCGT": "CCGT",
    "OCGT": "OCGT",
    "biomass": "biomass",
    "coal": "all_coal",
    "offwind": "wind_offshore",
    "oil": "oil",
    "onwind": "wind_onshore",
    "solar": "solar_pv",
    "nuclear": "nuclear",
    "hydro": "hydro",
    # Technologies not directly mapped, but indirectly through scale_technologies_relative_to
    # "lignite": "all_coal",
    # "ror": "hydro",
}

map_general_to_remind = {
    "CCGT": ["ngcc", "ngccc", "gaschp"],
    "OCGT": ["ngt"],
    "biomass": ["biochp", "bioigcc", "bioigccc"],
    "all_coal": ["igcc", "igccc", "pc", "coalchp"],
    "nuclear": ["tnrs", "fnrs"],
    "oil": ["dot"],
    "solar_pv": ["spv"],
    "wind_offshore": ["windoff"],
    "wind_onshore": ["wind"],
    "hydro": ["hydro"],
}

# Technologies not present in REMIND, scale these technologies based on development
# of their reference technologies (preserve the ratio the technologies see in the
# original PyPSA-EUR cost data and apply that ratio to the new costs)

scale_technologies_relative_to = {
    "solar-rooftop": "solar",
    "solar-utility": "solar",
    "PHS": "hydro",
    "ror": "hydro",
    "lignite": "coal",
    "offwind-ac-connection-submarine": "offwind",
    "offwind-ac-connection-underground": "offwind",
    "offwind-ac-station": "offwind",
    "offwind-dc-connection-submarine": "offwind",
    "offwind-dc-connection-underground": "offwind",
    "offwind-dc-station": "offwind",
}


# Inverted maps required here
map_remind_to_general = {lv: k for k, v in map_general_to_remind.items() for lv in v}
map_general_to_pypsaeur = pd.DataFrame(
    {
        "PyPSA-EUR": map_pypsaeur_to_general.keys(),
        "general": map_pypsaeur_to_general.values(),
    }
)

if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = SimpleNamespace()

        snakemake.wildcards = {
            "year": "2040",
            "iteration": 1
        }

        snakemake.input = {
            "original_costs": "../data/costs_2040.csv",  # Reference costs used for ratios between technologies witout values from REMIND-EU
            "region_mapping": "../config/regionmapping_21_EU11.csv",
            "remind_data": "../resources/no_scenario/i1/REMIND2PyPSAEUR.gdx",
        }

        snakemake.output = {
            "costs": "../resources/no_scenario/i1/y2040/costs.csv",
        }
        
        snakemake.config = {"countries": ["DE", "FR", "PL"]}
    
    else:
        configure_logging(snakemake)
    
    # Create region mapping by loading the original mapping from REMIND-EU from file
    # and then mapping ISO 3166-1 alpha-3 country codes to PyPSA-EUR ISO 3166-1 alpha-2 country codes
    logger.info("Loading region mapping and adding Kosovo (KV) manually.")
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

    logger.info("Loading REMIND data...")
    remind_data = gt.Container(snakemake.input["remind_data"])

    # Read investment costs
    logger.info("... extracting investment costs")
    costs = remind_data["vm_costTeCapital"].records.rename(
        columns={
            "ttot": "year",
            "all_regi": "region",
            "all_te": "technology",
            "level": "value",
        }
    )
    costs = costs.loc[costs["year"] == snakemake.wildcards["year"]]
    costs["value"] *= 1e6  # Unit conversion from TUSD/TW to USD/MW
    costs["parameter"] = "investment"
    costs["unit"] = "USD/MW"
    costs = costs[["region", "technology", "parameter", "value", "unit"]]

    # Add discount rate
    logger.info("... extracting discount rate")
    discount_rate = remind_data["p_r"].records.rename(
        columns={
            "ttot": "year",
            "all_regi": "region",
        }
    )
    discount_rate = discount_rate.loc[discount_rate["year"] == snakemake.wildcards["year"]]
    discount_rate["parameter"] = "discount_rate"
    discount_rate["unit"] = "p.u."
    discount_rate = discount_rate[["region", "parameter", "value", "unit"]]

    # Add lifetime
    logger.info("... extracting lifetime")
    lifetime = (
        remind_data["pm_data"]
        .records.query("char == 'lifetime'")
        .rename(
            columns={
                "all_regi": "region",
                "all_te": "technology",
            }
        )
    )
    lifetime["parameter"] = "lifetime"
    lifetime["unit"] = "years"
    lifetime = lifetime[["region", "technology", "parameter", "value", "unit"]]

    # Add fixed and variable O&M
    logger.info("... extracting FOM")
    fom = (
        remind_data["pm_data"]
        .records.query("char == 'omf'")
        .rename(
            columns={
                "all_regi": "region",
                "all_te": "technology",
            }
        )
    )
    fom["value"] *= 100  # Unit conversion from p.u. to %
    fom["parameter"] = "FOM"
    fom["unit"] = "%/year"
    fom = fom[["region", "technology", "parameter", "value", "unit"]]
    
    # Add variable O&M
    logger.info("... extracting VOM")
    vom = (
        remind_data["pm_data"]
        .records.query("char == 'omv'")
        .rename(
            columns={
                "all_regi": "region",
                "all_te": "technology",
            }
        )
    )
    vom["value"] *= 1e6 / 8760  # Unit conversion from TUSD/TWa to USD/MWh
    vom["parameter"] = "VOM"
    vom[
        "unit"
    ] = "USD/MWh"  # TODO check whether per unit input or unit output ( should be per unit MWh_e output)
    vom = vom[["region", "technology", "parameter", "value", "unit"]]

    # Add CO2 intensities
    logger.info("... extracting CO2 intensities")
    co2_intensity = remind_data["fm_dataemiglob"].records.rename(
        columns={
            "all_te_2": "technology",
            "all_enty_0": "from_carrier",
            "all_enty_1": "to_carrier",
            "all_enty_3": "emission_type",
        }
    )
    co2_intensity = co2_intensity.loc[
        (co2_intensity["to_carrier"] == "seel") & (co2_intensity["emission_type"] == "co2")
    ]
    # Unit conversion from Gt_C/TWa to t_CO2/MWh
    co2_intensity["value"] *= 1e9 * ((2 * 16 + 12) / 12) / 8760 / 1e6
    co2_intensity["parameter"] = "CO2 intensity"
    co2_intensity["unit"] = "t_CO2/MWh_th"  # TODO check correct unit
    co2_intensity = co2_intensity.merge(
        pd.Series(costs["region"].unique(), name="region"), how="cross"
    )  # Add region to match columns with remaining data
    co2_intensity = co2_intensity[["region", "technology", "parameter", "value", "unit"]]

    # Efficiencies; separated into two different variables in REMIND (constant & year-dependent)
    logger.info("... extracting efficiencies")
    efficiency = pd.concat(
        [
            remind_data["pm_eta_conv"].records,
            remind_data["pm_dataeta"].records,
        ]
    ).rename(
        columns={
            "tall": "year",
            "all_regi": "region",
            "all_te": "technology",
        }
    )
    efficiency = efficiency.loc[efficiency["year"] == snakemake.wildcards["year"]]
    efficiency["parameter"] = "efficiency"
    efficiency["unit"] = "p.u."  # TODO check correct unit
    efficiency = efficiency[["region", "technology", "parameter", "value", "unit"]]

    # Fuel costs
    logger.info("... extracting fuel costs")
    fuel_costs = remind_data["pm_PEPrice"].records.rename(
        columns={
            "ttot": "year",
            "all_regi": "region",
            "all_enty": "carrier",
        }
    )
    fuel_costs = fuel_costs.loc[fuel_costs["year"] == snakemake.wildcards["year"]]
    # Unit conversion from TUSD/TWa to USD/MWh
    fuel_costs["value"] *= 1e6 / 8760

    # Mapping for primary energy carriers to technologies for electricity generation
    map_carrier_technology = (
        remind_data["pe2se"]
        .records.query("all_enty_1 == 'seel'")
        .rename(
            columns={
                "all_enty_0": "carrier",
                "all_te_2": "technology",
            }
        )
        .set_index("carrier")["technology"]
    )

    fuel_costs = fuel_costs.merge(map_carrier_technology, on="carrier")
    fuel_costs["parameter"] = "fuel"
    fuel_costs["unit"] = "USD/MWh_th"  # TODO check correct unit (should be per MWh_th input)
    fuel_costs = fuel_costs[["region", "technology", "parameter", "value", "unit"]]

    # Special treatment for nuclear:
    # * Fuel costs are given in TUSD/Mt, which is converted to USD/MWh_el using the efficiency
    # * Efficiencies are given in TWa/Mt uranium, which we already apply to the fuel costs, thus set efficiency to 1
    fuel_costs = fuel_costs.set_index(["technology", "region"])
    efficiency = efficiency.set_index(["technology", "region"])
    fuel_costs.loc[["fnrs","tnrs"], "value"] /= (efficiency.loc[["fnrs","tnrs"]]["value"])
    fuel_costs.loc[["fnrs","tnrs"], "unit"] = "USD/MWh_el"
    efficiency.loc[["fnrs","tnrs"], "value"] = 1
    fuel_costs = fuel_costs.reset_index()
    efficiency = efficiency.reset_index()
    
    
    logger.info("Calculating weighted technology costs...")
    weights = remind_data["v32_usableSeTeDisp"].records.rename(
        columns={
            "ttot": "year",
            "all_regi": "region",
            "all_enty": "carrier",
            "all_te": "technology",
            "level": "value",
        }
    )
    weights = weights.loc[
        (weights["year"] == snakemake.wildcards["year"])
        & (weights["carrier"] == "seel")
        & (weights["region"].isin(region_mapping["REMIND-EU"]))
    ]

    weights["general_technology"] = weights["technology"].map(map_remind_to_general)

    # Calculate weight per technology and region based on aggregated general technology
    weights["weight"] = weights["value"].div(
        weights.groupby(weights["general_technology"])["value"].transform("sum")
    )
    # %%
    # Combine all parameters before weighting, remove irrelevant values from outside REMIND regions
    df = pd.concat([costs, lifetime, fom, vom, co2_intensity, efficiency, fuel_costs])
    df["general_technology"] = df["technology"].map(map_remind_to_general)
    df = df.loc[
        (df["region"].isin(weights["region"].unique())) & df["general_technology"].notnull()
    ]
    # %%
    # weighted aggregation with weights per region and general technology
    df["weight"] = df.apply(
        lambda x: weights.loc[
            (weights["region"] == x["region"]) & (weights["technology"] == x["technology"]),
            "weight",
        ].values[0],
        axis="columns",
    )
    df["weighted_value"] = df["value"] * df["weight"]
    df = (
        df.groupby(["general_technology", "parameter"])
        .agg(
            {
                "parameter": "first",
                "weighted_value": np.sum,
                "unit": pd.unique,  # Keep all unique units
            }
        )
        .rename(columns={"weighted_value": "value"})
    )
    # %%
    # Map general technologies to PyPSA-EUR technologies
    df = df.merge(map_general_to_pypsaeur, left_on="general_technology", right_on="general")
    df = df.rename(columns={"PyPSA-EUR": "technology"})[
        ["technology", "parameter", "value", "unit"]
    ]
    df["source"] = "REMIND-EU"
    df[
        "further description"
    ] = "Extracted from REMIND-EU model in 'import_REMIND_costs.py' script"

    # Overwrite original cost data with REMIND extracted cost data
    # Keep original values for data which is not available in REMIND
    df_base = pd.read_csv(snakemake.input["original_costs"]).set_index(
        ["technology", "parameter"]
    )
    df_base["original_value"] = df_base[
        "value"
    ]  # use this later to scale some technologies
    df_base.update(df.set_index(["technology", "parameter"]))

    df_base = df_base.reset_index()


    # %%
    logger.info("Scaling costs for technologies not extracted from REMIND-EU...")
    # Scale costs for technologies which are not extracted from REMIND based on reference technologies
    # and their ratio to those technologies (basically allow for indirect learning for derived/related technologies)
    def calculate_scaled_value(row):
        if row["technology"] in scale_technologies_relative_to:
            reference_row = df_base.loc[
                (df_base["technology"] == scale_technologies_relative_to[row["technology"]])
                & (df_base["parameter"] == row["parameter"])
            ].squeeze()

            if not reference_row.empty:
                row["value"] = (
                    reference_row["value"]
                    * row["original_value"]
                    / reference_row["original_value"]
                )
                row["unit"] = reference_row["unit"]
                row["source"] = "REMIND-EU"
                row["further description"] = "Scaled value from REMIND-EU model"

        return row


    df_base = df_base.apply(calculate_scaled_value, axis="columns")

    # Check values for plausibility
    # Efficiencies should be between 0 and 1
    if not efficiency["value"].between(0, 1).all():
        logger.warning("Efficiency values below 0 or above 1 detected.")

    # FOM in percent should be between 0 and 100
    if not df_base.query("parameter == 'FOM'")["value"].between(0, 100).all():
        logger.warning("Fixed O&M values below 0 or above 100% detected.")

    # VOM, investment, CO2 intensity and fuel cost should always be positive
    if (
        df_base.query("parameter in ['VOM', 'investment', 'CO2 intensity', 'fuel']")[
            "value"
        ]
        .lt(0)
        .any()
    ):
        logger.warning(
            "Negative values detected for VOM, investment, CO2 intensity or fuel."
        )

    # Lifetime should be between 10 and 100 years (guess)
    if not df_base.query("parameter == 'lifetime'")["value"].between(10, 100).all():
        logger.warning("Lifetime values below 10 or above 100 years detected.")

    # %%
    # Write results to file
    df_base.to_csv(snakemake.output["costs"], index=False)