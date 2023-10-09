# -*- coding: utf-8 -*-
# %%
import logging
from types import SimpleNamespace

import country_converter as coco
import numpy as np
import pandas as pd
from _helpers import (
    configure_logging,
    get_region_mapping,
    get_technology_mapping,
    read_remind_data,
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "import_REMIND_costs",
            year="2055",
            iteration="1",
            scenario="PyPSA_NPi_preFacAuto_Avg_preFacFadeOut_adjCost_2023-10-09_19.43.41",
        )

    configure_logging(snakemake)

    # Only Generation technologies (PyPSA "generator" "carriers")
    # Use a two step mapping approach between PyPSA-EUR and REMIND:
    # First mapping is aggregating PyPSA-EUR technologies to general technologies
    # Second mapping is disaggregating general technologies to REMIND technologies
    map_pypsaeur_to_general = get_technology_mapping(
        snakemake.input["technology_mapping"],
        source="PyPSA-EUR",
        target="General",
        flatten=True,
    )

    # Remove elements with keys:
    # offwind-ac, offwind-dc: Technologies already mapped through "offwind"
    # lignite, ror: Technologies not directly mapped, but indirectly through scale_technologies_relative_to
    map_pypsaeur_to_general = {
        k: v
        for k, v in map_pypsaeur_to_general.items()
        if k not in ["offwind-ac", "offwind-dc", "ror", "lignite"]
    }

    map_general_to_remind = get_technology_mapping(
        snakemake.input["technology_mapping"], source="General", target="REMIND-EU"
    )

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
    map_remind_to_general = {
        lv: k for k, v in map_general_to_remind.items() for lv in v
    }
    map_general_to_pypsaeur = pd.DataFrame(
        {
            "PyPSA-EUR": map_pypsaeur_to_general.keys(),
            "general": map_pypsaeur_to_general.values(),
        }
    )

    # Load region mapping
    region_mapping = pd.DataFrame(
        get_region_mapping(
            snakemake.input["region_mapping"], source="PyPSA-EUR", target="REMIND-EU"
        )
    ).T.reset_index()
    region_mapping.columns = ["PyPSA-EUR", "REMIND-EU"]
    # Limit mapping to countries modelled with PyPSA-EUR
    region_mapping = region_mapping.loc[
        region_mapping["PyPSA-EUR"].isin(snakemake.config["countries"])
    ]

    # Read investment costs
    logger.info("... extracting investment costs")
    costs = read_remind_data(
        file_path=snakemake.input["remind_data"],
        variable_name="p32_capCostwAdjCost",
        rename_columns={
            "ttot": "year",
            "all_regi": "region",
            "all_te": "technology",
        },
    )
    costs = costs.loc[costs["year"] == snakemake.wildcards["year"]]
    costs["value"] *= 1e6  # Unit conversion from TUSD/TW to USD/MW
    costs["parameter"] = "investment"
    costs["unit"] = "USD/MW"
    costs = costs[["region", "technology", "parameter", "value", "unit"]]

    # Add discount rate
    logger.info("... extracting discount rate")
    discount_rate = read_remind_data(
        file_path=snakemake.input["remind_data"],
        variable_name="p_r",
        rename_columns={
            "ttot": "year",
            "all_regi": "region",
        },
    )
    discount_rate = discount_rate.loc[
        discount_rate["year"] == snakemake.wildcards["year"]
    ]
    discount_rate["parameter"] = "discount_rate"
    discount_rate["unit"] = "p.u."
    discount_rate = discount_rate[["region", "parameter", "value", "unit"]]

    # Add lifetime
    logger.info("... extracting lifetime")
    lifetime = read_remind_data(
        file_path=snakemake.input["remind_data"],
        variable_name="pm_data",
        rename_columns={
            "all_regi": "region",
            "all_te": "technology",
        },
    ).query("char == 'lifetime'")
    lifetime["parameter"] = "lifetime"
    lifetime["unit"] = "years"
    lifetime = lifetime[["region", "technology", "parameter", "value", "unit"]]

    # Add fixed and variable O&M
    logger.info("... extracting FOM")
    fom = read_remind_data(
        file_path=snakemake.input["remind_data"],
        variable_name="pm_data",
        rename_columns={
            "all_regi": "region",
            "all_te": "technology",
        },
    ).query("char == 'omf'")
    fom["value"] *= 100  # Unit conversion from p.u. to %
    fom["parameter"] = "FOM"
    fom["unit"] = "%/year"
    fom = fom[["region", "technology", "parameter", "value", "unit"]]

    # Add variable O&M
    logger.info("... extracting VOM")
    vom = read_remind_data(
        file_path=snakemake.input["remind_data"],
        variable_name="pm_data",
        rename_columns={
            "all_regi": "region",
            "all_te": "technology",
        },
    ).query("char == 'omv'")
    vom["value"] *= 1e6 / 8760  # Unit conversion from TUSD/TWa to USD/MWh
    vom["parameter"] = "VOM"
    vom[
        "unit"
    ] = "USD/MWh"  # TODO check whether per unit input or unit output ( should be per unit MWh_e output)
    vom = vom[["region", "technology", "parameter", "value", "unit"]]

    # Add CO2 intensities
    logger.info("... extracting CO2 intensities")
    co2_intensity = read_remind_data(
        file_path=snakemake.input["remind_data"],
        variable_name="fm_dataemiglob",
        rename_columns={
            "all_te_2": "technology",
            "all_enty_0": "from_carrier",
            "all_enty_1": "to_carrier",
            "all_enty_3": "emission_type",
        },
    )

    co2_intensity = co2_intensity.loc[
        (co2_intensity["to_carrier"] == "seel")
        & (co2_intensity["emission_type"] == "co2")
    ]
    # Unit conversion from Gt_C/TWa to t_CO2/MWh
    co2_intensity["value"] *= 1e9 * ((2 * 16 + 12) / 12) / 8760 / 1e6
    co2_intensity["parameter"] = "CO2 intensity"
    co2_intensity["unit"] = "t_CO2/MWh_th"  # TODO check correct unit
    co2_intensity = co2_intensity.merge(
        pd.Series(costs["region"].unique(), name="region"), how="cross"
    )  # Add region to match columns with remaining data
    co2_intensity = co2_intensity[
        ["region", "technology", "parameter", "value", "unit"]
    ]

    # Efficiencies; separated into two different variables in REMIND (constant & year-dependent)
    logger.info("... extracting efficiencies")
    pm_eta_conv = read_remind_data(
        file_path=snakemake.input["remind_data"],
        variable_name="pm_eta_conv",
        rename_columns={
            "tall": "year",
            "all_regi": "region",
            "all_te": "technology",
        },
    )
    pm_dataeta = read_remind_data(
        file_path=snakemake.input["remind_data"],
        variable_name="pm_dataeta",
        rename_columns={
            "tall": "year",
            "all_regi": "region",
            "all_te": "technology",
        },
    )
    efficiency = pd.concat([pm_eta_conv, pm_dataeta])
    efficiency = efficiency.loc[efficiency["year"] == snakemake.wildcards["year"]]
    efficiency["parameter"] = "efficiency"
    efficiency["unit"] = "p.u."  # TODO check correct unit
    efficiency = efficiency[["region", "technology", "parameter", "value", "unit"]]

    # Fuel costs
    logger.info("... extracting fuel costs")
    fuel_costs = read_remind_data(
        file_path=snakemake.input["remind_data"],
        variable_name="p32_PEPriceAvg",
        rename_columns={
            "ttot": "year",
            "all_regi": "region",
            "all_enty": "carrier",
        },
    )
    fuel_costs = fuel_costs.loc[fuel_costs["year"] == snakemake.wildcards["year"]]
    # Unit conversion from TUSD/TWa to USD/MWh
    fuel_costs["value"] *= 1e6 / 8760

    # Mapping for primary energy carriers to technologies for electricity generation
    map_carrier_technology = (
        read_remind_data(
            file_path=snakemake.input["remind_data"],
            variable_name="pe2se",
            rename_columns={
                "all_enty_0": "carrier",
                "all_te_2": "technology",
            },
        )
        .query("all_enty_1 == 'seel'")
        .set_index("carrier")["technology"]
    )

    fuel_costs = fuel_costs.merge(map_carrier_technology, on="carrier")
    fuel_costs["parameter"] = "fuel"
    fuel_costs[
        "unit"
    ] = "USD/MWh_th"  # TODO check correct unit (should be per MWh_th input)
    fuel_costs = fuel_costs[["region", "technology", "parameter", "value", "unit"]]

    # Special treatment for nuclear:
    # * Fuel costs are given in TUSD/Mt, which is converted to USD/MWh_el using the efficiency
    # * Efficiencies are given in TWa/Mt uranium, which we already apply to the fuel costs, thus set efficiency to 1
    fuel_costs = fuel_costs.set_index(["technology", "region"])
    efficiency = efficiency.set_index(["technology", "region"])
    fuel_costs.loc[["fnrs", "tnrs"], "value"] /= efficiency.loc[["fnrs", "tnrs"]][
        "value"
    ]
    fuel_costs.loc[["fnrs", "tnrs"], "unit"] = "USD/MWh_el"
    efficiency.loc[["fnrs", "tnrs"], "value"] = 1
    fuel_costs = fuel_costs.reset_index()
    efficiency = efficiency.reset_index()

    logger.info("Calculating weighted technology costs...")
    weights = read_remind_data(
        file_path=snakemake.input["remind_data"],
        variable_name="v32_usableSeTeDisp",
        rename_columns={
            "ttot": "year",
            "all_regi": "region",
            "all_enty": "carrier",
            "all_te": "technology",
            "level": "value",
        },
    ).query(
        "carrier == 'seel' and region.isin(@region_mapping['REMIND-EU'])",
        engine="python",
    )[
        ["year", "region", "technology", "value"]
    ]
    weights["general_technology"] = weights["technology"].map(map_remind_to_general)

    # Calculate weight per technology and region based on aggregated general technology
    weights = weights.merge(
        weights.groupby(["year", "region", "general_technology"])["value"]
        .sum()
        .rename("general_technology_sum"),
        on=["year", "region", "general_technology"],
    )
    # set very small values to nan (prevent weird weights based on very small total capacities)
    weights["general_technology_sum"] = weights["general_technology_sum"].where(
        lambda x: x > 1e-4
    )
    weights["weight"] = weights["value"] / weights["general_technology_sum"]

    # If non of the technologies of a general_technology are built, the weight becomes NaN which
    # we want to avoid, else costs could be NaN or 0.
    # Instead forward and backward fill the values with the weight of the previous / next time step year
    weights["weight"] = weights.groupby(["year", "region", "technology"])[
        "weight"
    ].ffill()
    weights["weight"] = weights.groupby(["year", "region", "technology"])[
        "weight"
    ].bfill()

    if weights["weight"].isna().any():
        # In rare cases, where none of the technologies of a general_technology is built in *any* previous year,
        # a backup weight is calculated as 1 / number of technologies in the general_technology
        # and then assigned to fill the NaN values
        backup_weights = (
            1
            / weights.groupby(["general_technology", "year", "region"])["value"].count()
        ).to_frame("backup_weight")
        weights = weights.set_index(["year", "region", "general_technology"]).join(
            backup_weights
        )
        weights["weight"] = weights["weight"].fillna(weights["backup_weight"])
        weights = weights[["technology", "weight"]].reset_index()

        # To future editor: Remove this warning if you disagree
        logging.warning(
            "NaN values in weights were filled with backup weights. This is should not happen very often and is probably a mistake."
        )

    # After filling NaN values, now limit weights to the year of interest
    weights = weights.loc[(weights["year"] == snakemake.wildcards["year"])]

    # Combine all parameters before weighting, remove irrelevant values from outside REMIND regions
    df = pd.concat([costs, lifetime, fom, vom, co2_intensity, efficiency, fuel_costs])

    # For each region and technology pair, add the region-specific discount_rate as an additional row
    df = pd.concat(
        [
            df,
            df[["region", "technology"]]
            .merge(discount_rate, on="region")
            .drop_duplicates(),
        ]
    )

    df["general_technology"] = df["technology"].map(map_remind_to_general)
    # Limit to region of interest and only consider technologies with weights
    df = df.query(
        "region.isin(@weights.region.unique()) and general_technology.notnull()"
    )

    # %%
    # Adds weights to df for weighted aggregation with weights per region and general technology
    df = df.merge(
        weights[["region", "technology", "weight"]], on=["region", "technology"]
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

    # check if there are multiple units per parameter
    if len(df) != len(df.explode("unit")):
        problematic_technologies = (
            df["unit"].apply(len).where(lambda x: x != 1).dropna().index
        )
        logger.warning(
            f"Multiple units per parameter detected. Please check: {problematic_technologies}"
        )
    # Turn unit column entries from lists caused by pd.unique to strings
    df = df.explode("unit")

    # Map general technologies to PyPSA-EUR technologies
    df = df.merge(
        map_general_to_pypsaeur, left_on="general_technology", right_on="general"
    )
    df = df.rename(columns={"PyPSA-EUR": "technology"})[
        ["technology", "parameter", "value", "unit"]
    ]
    df["source"] = "REMIND-EU"
    df[
        "further description"
    ] = "Extracted from REMIND-EU model in 'import_REMIND_costs.py' script"

    # Overwrite original cost data with REMIND extracted cost data
    # Keep original values for data which is not available in REMIND
    # and add new values from REMIND which have previously not existed (e.g. discount rate)
    df_base = pd.read_csv(snakemake.input["original_costs"]).set_index(
        ["technology", "parameter"]
    )
    df_base["original_value"] = df_base[
        "value"
    ]  # use this later to scale some technologies
    df_base = (
        df.set_index(["technology", "parameter"]).combine_first(df_base).reset_index()
    )

    logger.info("Scaling costs for technologies not extracted from REMIND-EU...")

    # Scale costs for technologies which are not extracted from REMIND based on reference technologies
    # and their ratio to those technologies (basically allow for indirect learning for derived/related technologies)
    def calculate_scaled_value(row):
        if row["technology"] in scale_technologies_relative_to:
            reference_row = df_base.loc[
                (
                    df_base["technology"]
                    == scale_technologies_relative_to[row["technology"]]
                )
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

    # discount_rate must always be greater 0
    if not df_base.query("parameter == 'discount_rate'")["value"].gt(0).all():
        logger.warning("discount rate values <= 0 detected.")

    # discount rate, VOM, investment, CO2 intensity and fuel cost should always be positive
    if (
        df_base.query(
            "parameter in ['discount_rate', 'VOM', 'investment', 'CO2 intensity', 'fuel']"
        )["value"]
        .lt(0)
        .any()
    ):
        logger.warning(
            "Negative values detected for discount rate, VOM, investment, CO2 intensity or fuel."
        )

    # Lifetime should be between 10 and 100 years (guess)
    if not df_base.query("parameter == 'lifetime'")["value"].between(10, 100).all():
        logger.warning("Lifetime values below 10 or above 100 years detected.")

    # No values should be negative (except for CO2 intensity which can be negative)
    if df_base.query("parameter != 'CO2 intensity'")["value"].lt(0).any():
        raise ValueError(
            f"Negative values detected for:\n" f"{df_base.loc[df_base['value'].lt(0)]}"
        )

    # Write results to file
    df_base.to_csv(snakemake.output["costs"], index=False)
