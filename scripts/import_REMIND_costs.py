# -*- coding: utf-8 -*-
# %%
# There are three steps for importing REMIND-EU costs followed in this script:
# 1. Technologies which get their values from REMIND-EU, weighted by the electricity generation of the related REMIND-EU technology
# 2. Technologies where values are scaled based on a proxy technology
# 3. Technologies where values are set in the technology mapping config file
# 4. Add discount rate for all technologies where not discount rate is set in the technology mapping config file

import logging

import pandas as pd
import yaml
from _helpers import configure_logging, get_region_mapping, read_remind_data

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "import_REMIND_costs",
            scenario="PyPSA_NPi_preFacAuto_Avg_2023-11-10_09.42.27",
            iteration="1",
            year="2025",
        )

    configure_logging(snakemake)

# Load region mapping
logger.info("Loading region mapping ... ")
region_mapping = (
    pd.DataFrame.from_dict(
        get_region_mapping(
            snakemake.input["region_mapping"],
            source="PyPSA-EUR",
            target="REMIND-EU",
        )
    )
    .T.reset_index()
    .rename(columns={"index": "PyPSA-EUR", 0: "REMIND-EU"})
)

# Limit to regions & countries PyPSA-EUR is configured for
region_mapping = region_mapping.query(
    f"`PyPSA-EUR`.isin({snakemake.config['countries']})"
)

# Read new technology mapping
logger.info("Loading technology mapping ... ")
technology_mapping = pd.read_csv(
    snakemake.input["technology_cost_mapping"],
)

# Convert list-like entries to real lists and explode for 1:1 mapping per row entry between PyPSA-EUR and REMIND-EU technologies
technology_mapping["reference"] = (
    technology_mapping["reference"].apply(yaml.safe_load).to_list()
)
technology_mapping = technology_mapping.explode("reference")

# +++ 1. Technologies which get their values from REMIND-EU, weighted by the electricity generation of the related REMIND-EU technology +++
mapped_technologies = technology_mapping.query(
    "`couple to` == 'mapping generation weighted to reference REMIND-EU technology'"
).drop(columns=["unit"])

## Load REMIND data into long format
# investment costs
logger.info("... extracting investment costs")
costs = read_remind_data(
    file_path=snakemake.input["remind_data"],
    variable_name="p32_capCostwAdjCost",
    rename_columns={
        "ttot": "year",
        "all_regi": "region",
        "all_te": "technology",
    },
).query("year == '{}'".format(snakemake.wildcards["year"]))
costs["value"] *= 1e6  # Unit conversion from TUSD/TW to USD/MW
costs["parameter"] = "investment"
costs["unit"] = "USD/MW"

# lifetime
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

# fixed O&M
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

# variable O&M
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

# CO2 intensities
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
).query("to_carrier == 'seel' & emission_type == 'co2'")

# CO2 intensities are not region specified; instead of treating as special case, continue by
# adding a fake region-dependency with all-region identical values
co2_intensity = co2_intensity.merge(
    pd.Series(costs["region"].unique(), name="region"), how="cross"
)
# Unit conversion from Gt_C/TWa to t_CO2/MWh
co2_intensity["value"] *= 1e9 * ((2 * 16 + 12) / 12) / 8760 / 1e6
co2_intensity["parameter"] = "CO2 intensity"
co2_intensity["unit"] = "t_CO2/MWh_th"  # TODO check correct unit

# Efficiencies
# Values are split across two different variables in REMIND (constant & year-dependent)
logger.info("... extracting efficiencies")
efficiency = pd.concat(
    [
        read_remind_data(
            file_path=snakemake.input["remind_data"],
            variable_name="pm_eta_conv",
            rename_columns={
                "tall": "year",
                "all_regi": "region",
                "all_te": "technology",
            },
        ),
        read_remind_data(
            file_path=snakemake.input["remind_data"],
            variable_name="pm_dataeta",
            rename_columns={
                "tall": "year",
                "all_regi": "region",
                "all_te": "technology",
            },
        ),
    ]
).query("year == '{}'".format(snakemake.wildcards["year"]))
efficiency["parameter"] = "efficiency"
efficiency["unit"] = "p.u."  # TODO check correct unit
# Special treatment for nuclear: Efficiencies are in TWa/Mt=8760 TWh/Tg_U -> convert to MWh/g_U to match with fuel costs in USD/g_U
efficiency.loc[efficiency["technology"].isin(["fnrs", "tnrs"]), "value"] *= 8760 / 1e6
efficiency.loc[efficiency["technology"].isin(["fnrs", "tnrs"]), "unit"] = "MWh/g_U"


# Fuel costs
logger.info("... extracting fuel costs")
fuel_costs = read_remind_data(
    file_path=snakemake.input["remind_data"],
    variable_name="p32_PEPriceAvg",
    rename_columns={
        "ttot": "year",
        "all_regi": "region",
        "all_enty": "technology",
    },
).query("year == '{}'".format(snakemake.wildcards["year"]))
fuel_costs["parameter"] = "fuel"
# Unit conversion from TUSD/TWa to USD/MWh
# Special treatment for nuclear fuel uranium (peur): Fuel costs are originally in TUSD/Mt = USD/g_U (TUSD/Tg) -> adjust unit
fuel_costs.loc[~(fuel_costs["technology"] == "peur"), "value"] *= 1e6 / 8760
fuel_costs[
    "unit"
] = "USD/MWh_th"  # TODO check correct unit (should be per MWh_th input)
fuel_costs.loc[fuel_costs["technology"] == "peur", "unit"] = "USD/g_U"

# Combine all technology data for further processing
df = pd.concat([costs, lifetime, fom, vom, co2_intensity, efficiency, fuel_costs])[
    ["region", "technology", "parameter", "value", "unit"]
].rename(columns={"technology": "reference"})

# Limit to regions & countries REMIND-EU is configured for
df = df.query("region.isin(@region_mapping['REMIND-EU'])", engine="python")

# To calculate weighted values for technologies / other cost related parameters,
# first load the weights calculated in REMIND-EU
weights = pd.concat(
    [
        read_remind_data(
            file_path=snakemake.input["remind_data"],
            variable_name="p32_weightGen",
            rename_columns={
                "ttot": "year",
                "all_regi": "region",
                "all_te": "technology",
                "value": "weight",
            },
        ),
        read_remind_data(
            file_path=snakemake.input["remind_data"],
            variable_name="p32_weightPEprice",
            rename_columns={
                "ttot": "year",
                "all_regi": "region",
                "all_enty": "technology",
                "value": "weight",
            },
        ),
        read_remind_data(
            file_path=snakemake.input["remind_data"],
            variable_name="p32_weightStor",
            rename_columns={
                "ttot": "year",
                "all_regi": "region",
                "all_te": "technology",
                "value": "weight",
            },
        ),
    ]
).query(
    "`region`.isin(@region_mapping['REMIND-EU']) and year == '{}'".format(
        snakemake.wildcards["year"]
    ),
    engine="python",
)[
    ["region", "technology", "weight"]
]

# Merge weights to REMIND-EU technology data
df = df.merge(
    weights,
    left_on=["region", "reference"],
    right_on=["region", "technology"],
    how="left",
)[["region", "reference", "parameter", "value", "unit", "weight"]]


# Create new technology data for all technologies which are mapped and generation weighted
mapped_technologies = mapped_technologies.merge(
    df, on=["reference", "parameter"], how="left"
)

# Some parameters are not reported by REMIND-EU if they are not build, e.g. efficiency.
# After merge these values will be NaN, fill here with 0
mapped_technologies["weight"] = mapped_technologies["weight"].fillna(0.0)

# Generation is reported in TWa and reported as 0 for technologies which are not built;
# by adding a small value, we can avoid NaN values in the weighted aggregation
# Value added is 1 MWh = 1e-6/8760 TWa
mapped_technologies["weight"] += 1e-6 / 8760


# Helper function used together with pd.DataFrame.apply:
# * Calculate weighted value per technology and parameter
# * Determine unit (which should be identical for all aggregated values)
def calculate_weighted_value(x):
    result = {
        "weighted_value": (x["value"] * x["weight"]).sum(skipna=False)
        / x["weight"].sum(),
        "unit": x["unit"].unique()[0],
    }

    assert (
        len(x["unit"].unique()) == 1
    ), f'Multiple units per parameter detected. Check: {x[["PyPSA-EUR technology", "parameter"]]}'

    return pd.Series(result, index=["weighted_value", "unit"])


# Calculate electricity-generation weighted technology parameter (weighted across regions and different REMIND-EU technologies per single PyPSA-EUR technology)
mapped_technologies = mapped_technologies.groupby(
    ["PyPSA-EUR technology", "parameter"]
).apply(calculate_weighted_value)
mapped_technologies = mapped_technologies.reset_index().rename(
    columns={"weighted_value": "value", "PyPSA-EUR technology": "technology"}
)
mapped_technologies["source"] = "REMIND-EU"
mapped_technologies[
    "further description"
] = "Extracted from REMIND-EU model in 'import_REMIND_costs.py' script"


# +++ 2. Technologies where values are scaled based on a proxy technology +++
scaled_technologies = technology_mapping.query(
    "`couple to` == 'scaling original values based on reference PyPSA-EUR technology'"
).drop(columns=["unit"])

# Check if technologies can be scaled
if not (
    missing_technologies := scaled_technologies.loc[
        ~scaled_technologies["reference"].isin(
            mapped_technologies["technology"].unique()
        )
    ]
).empty:
    raise AssertionError(
        f"Scaling only works if the technogloy used as reference is mapped from a REMIND-EU technology."
        f"The following technologies do not use reference technology which was previously mapped to REMIND-EU technologies."
        f"Check: {missing_technologies}"
    )

# Load original costs used by standard PyPSA-EUR model used as calculation basis
original_cost = pd.read_csv(snakemake.input["original_costs"])

# Get information on the reference technology (original cost in standard PyPSA-EUR) as well as the technologies REMIND-EU value
df = scaled_technologies.merge(
    mapped_technologies,
    left_on=["reference", "parameter"],
    right_on=["technology", "parameter"],
    how="left",
    validate="many_to_one",
).rename(columns={"value": "reference_value_new", "unit": "reference_unit_new"})

# Add original PyPSA-EUR technology assumptions which will then be scaled
df = df.merge(
    original_cost,
    left_on=["technology", "parameter"],
    right_on=["technology", "parameter"],
    how="left",
    validate="many_to_one",
).rename(
    columns={"value": "reference_value_original", "unit": "reference_unit_original"}
)
# Calculate the scaling factor
df["scale_by"] = df["reference_value_new"] / df["reference_value_original"]
# Technologies reported in PyPSA-EUR with kW or kWh have a scale_factor off by 1000 which needs correction
df.loc[df["reference_unit_original"].str.contains("kW"), "scale_by"] /= 1000

# Get the original value from standard PyPSA-EUR for the technology to scale
df = df.merge(
    original_cost,
    left_on=["PyPSA-EUR technology", "parameter"],
    right_on=["technology", "parameter"],
    how="left",
    validate="one_to_one",
)

# Scale original value
df["new_value"] = df["value"] * df["scale_by"]

df.loc[df["unit"].str.contains("kW"), "new_value"] /= 1000
# Adjust unit: Unit has to be adjusted, because REMIND-EU uses MW and MWh, PyPSA-EUR sometimes kW and kWh as basis for scaling
df["new_unit"] = df["unit"].str.replace("kW", "MW").str.replace("EUR", "USD")

df["further description"] = (
    "Original value from PyPSA-EUR scaled by cost development experienced in REMIND-EU experienced by the reference technology: "
    + df["reference"]
    + "\n"
    "Original source: "
    + df["source_x"]
    + " with description: "
    + df["further description_x"]
)
df["source"] = "REMIND-EU and PyPSA-EUR"

scaled_technologies = df[
    [
        "PyPSA-EUR technology",
        "parameter",
        "new_value",
        "new_unit",
        "source",
        "further description",
    ]
].rename(
    columns={
        "PyPSA-EUR technology": "technology",
        "new_value": "value",
        "new_unit": "unit",
    }
)


# +++ 3. Technologies where values are set in the technology mapping config file +++
set_technologies = technology_mapping.query(
    "`couple to` == 'setting to reference value'"
).rename(
    columns={
        "PyPSA-EUR technology": "technology",
        "reference": "value",
        "comment": "further description",
    }
)[
    ["technology", "parameter", "value", "unit", "further description"]
]
set_technologies[
    "source"
] = f"Set via configuration file: {snakemake.input['technology_cost_mapping']}"
set_technologies["further description"] = set_technologies[
    "further description"
].fillna("")

# Combine all technologies
costs = pd.concat([mapped_technologies, scaled_technologies, set_technologies])

# +++ 4. Add discount rate +++
# Discount rate is calculated on REMIND-EU side and just needs to be added for all technologies
# By adding the discount rate after the 3. step, we allow the discount rate to be overwritten in the technology mapping config file

# Get technologies with and without "discount rate"
discount_rate_technologies = costs.loc[
    costs["parameter"] == "discount rate", "technology"
]
technologies_without_discount_rate = costs.loc[
    ~costs["technology"].isin(discount_rate_technologies)
]

logger.info("... extracting discount rate")
discount_rate = read_remind_data(
    file_path=snakemake.input["remind_data"],
    variable_name="p32_discountRate",
    rename_columns={
        "ttot": "year",
    },
).query("year == '{}'".format(snakemake.wildcards["year"]))

assert (
    discount_rate.shape[0] == 1
), "Multiple discount rates instead of a single value found"

# Construct dataframe with discount rate for all technologies by cartesian product
discount_rate = (
    pd.Series(
        {
            "parameter": "discount rate",
            "value": discount_rate["value"].item(),
            "unit": "p.u.",
            "source": "REMIND-EU",
            "further description": "p32_discountRate",
        }
    )
    .to_frame()
    .T
)
discount_rate = discount_rate.merge(
    technologies_without_discount_rate[["technology"]], how="cross"
)

# Add discount rate to costs
costs = pd.concat([costs, discount_rate])

# Output to file
costs.to_csv(snakemake.output["costs"], index=False)

# %%
# list all rows in r with nan values inside
assert costs[
    costs.isna().any(axis=1)
].empty, f"NaN values in costs detected: {costs[costs.isna().any(axis=1)]}"

# -> this is bad, can we have at least investment and VOM values for all technologies? from REMIND?
