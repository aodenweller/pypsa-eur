# -*- coding: utf-8 -*-
# %%
import logging
import re
from pathlib import Path

import country_converter as coco
import numpy as np
import pandas as pd
import pypsa
from _helpers import (
    configure_logging,
    get_region_mapping,
    get_technology_mapping,
    read_remind_data,
)
from gams import transfer as gt

logger = logging.getLogger(__name__)
if not "snakemake" in globals():
    from _helpers import mock_snakemake

    snakemake = mock_snakemake(
        "extract_coupling_parameters",
        configfiles="config.remind.yaml",
        iteration="1",
    )

    # mock_snakemake doesn't work with checkpoints
    input_networks = [
        f"../results/no_scenario/i1/y{year}/networks/elec_s_4_ec_lcopt_1H-RCL-Ep0.0.nc"
        for year in [
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
        ]
    ]
else:
    input_networks = snakemake.input["networks"]
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
map_pypsaeur_to_general.pop("offwind")  # not needed

map_general_to_remind = get_technology_mapping(
    snakemake.input["technology_mapping"], source="General", target="REMIND-EU"
)

map_pypsaeur_to_remind_loads = {
    "AC": ["AC"],
}

# Create region mapping
region_mapping = get_region_mapping(
    snakemake.input["region_mapping"], source="PyPSA-EUR", target="REMIND-EU"
)
region_mapping = pd.DataFrame(region_mapping).T.reset_index()
region_mapping.columns = ["PyPSA-EUR", "REMIND-EU"]
region_mapping = region_mapping.set_index("PyPSA-EUR")


def check_for_mapping_completeness(n):
    tmp_set = (
        set(n.generators["carrier"]) - map_pypsaeur_to_general.keys() - set(["load"])
    )
    if tmp_set:
        logger.info(
            f"Technologies (carriers) missing from mapping PyPSA-EUR -> general technologies:\n {tmp_set}"
        )

    tmp_set = map_pypsaeur_to_general.values() - map_general_to_remind.keys()
    if tmp_set:
        logger.info(
            f"Technologies (carriers) missing from mapping General -> REMIND-EU:\n {tmp_set}"
        )

    tmp_set = set(n.loads["bus_carrier"]) - map_pypsaeur_to_remind_loads.keys()
    if tmp_set:
        logger.info(
            f"Technologies (carriers) missing from mapping PyPSA-EUR -> REMIND-EU (loads):\n {tmp_set}"
        )

    tmp_set = set(n.buses["country"]) - set(region_mapping.index)
    if tmp_set:
        logger.info(
            f"PyPSA-EUR countries without mapping to REMIND-EU regions::\n {tmp_set}"
        )


capacity_factors = []
generation_shares = []
installed_capacities = []
market_values = []
electricity_prices = []

for fp in input_networks:
    # Extract year from filename, format: elec_y<YYYY>_<morestuff>.nc
    m = re.findall(r"y(\d{4})", fp)
    assert len(m) == 1, "Unable to extract year from network path"
    year = int(m[0])

    # Load network
    network = pypsa.Network(fp)

    # First map the PyPSA-EUR countries to REMIND-EU regions;
    # .statistics(..) can then automatically take care of the aggregation
    network.buses["region"] = network.buses["country"].map(region_mapping["REMIND-EU"])

    # Add information for aggregation later: region name (REMIND-EU) and general technology
    network.generators["region"] = network.generators["bus"].map(
        network.buses["region"]
    )
    network.generators["general_carrier"] = network.generators["carrier"].map(
        map_pypsaeur_to_general
    )
    network.loads["region"] = network.loads["bus"].map(network.buses["region"])
    network.loads["bus_carrier"] = network.loads["bus"].map(network.buses["carrier"])

    # Now make sure we have all carriers in the mapping
    check_for_mapping_completeness(network)

    ## Extract coupling parameters from network

    # Calculate capacity factors; by assigning the general carrier and grouping by here, the capacity factor is automatically
    # calculated across all "carrier" technologies that map to the same "general_carrier"
    capacity_factor = network.statistics(
        comps=["Generator"], groupby=["region", "general_carrier"]
    )["Capacity Factor"]
    capacity_factor = capacity_factor.to_frame("value").reset_index()
    capacity_factor["year"] = year
    capacity_factors.append(capacity_factor)

    # Calculate shares of technologies in annual generation
    generation_share = (
        network.statistics.supply(
            comps=["Generator"],
            groupby=["region", "general_carrier"],
        )
        / network.statistics.supply(
            comps=["Generator"],
            groupby=["region", "general_carrier"],
        ).sum()
    )
    generation_share = generation_share.to_frame("value").reset_index()
    generation_share["year"] = year
    generation_shares.append(generation_share)

    # Calculate technology installed capacities
    installed_capacity = network.statistics(
        comps=["Generator"], groupby=["region", "general_carrier"]
    )["Optimal Capacity"]
    installed_capacity = installed_capacity.to_frame("value").reset_index()
    installed_capacity["year"] = year
    installed_capacities.append(installed_capacity)

    # Calculate the market values (round-about way as the intended method of the statistics module is not yet available)
    market_value = network.statistics.market_value(
        comps=["Generator"], groupby=["region", "general_carrier"]
    )
    market_value = market_value.to_frame("value").reset_index()
    market_value["year"] = year
    market_values.append(market_value)

    # Calculate load-weighted electricity prices based on bus marginal prices
    electricity_price = network.statistics.revenue(
        comps=["Load"], groupby=["region", "bus_carrier"]
    ) / (
        -1
        * network.statistics.withdrawal(
            comps=["Load"], groupby=["region", "bus_carrier"]
        )
    )
    electricity_price = electricity_price.to_frame("value").reset_index()
    electricity_price["year"] = year
    electricity_prices.append(electricity_price)


## Combine DataFrames to same format
# Helper function
def postprocess_dataframe(df):
    """
    General function to postprocess the dataframes, combines the network-
    specific results into one dataframe removes excess columns / sets index and
    sorts by region + year.
    """
    df = pd.concat(df)
    df = df.rename(
        columns={"general_carrier": "carrier", "bus_carrier": "carrier"}
    )  # different auxiliary columns have different names; rename for consistency

    df = df.set_index(
        ["year", "region", "carrier"]
    ).sort_index()  # set and sort by index for more logical sort order

    df = df[["value"]]
    df = df.reset_index()

    def map_carriers_for_remind(dg):
        """
        Maps the carrier names from general technologies to REMIND
        technologies.
        """
        old_carrier = dg.iloc[0]["carrier"]

        # Mapping for generators and loads are different
        if old_carrier in map_general_to_remind.keys():
            _map = map_general_to_remind
        elif old_carrier in map_pypsaeur_to_remind_loads.keys():
            _map = map_pypsaeur_to_remind_loads

        new_carriers = _map[old_carrier]

        # Repeat rows for each new carrier, create new dataframe then assign the new carrier name
        dg = pd.DataFrame(
            np.repeat(dg.values, len(new_carriers), axis=0), columns=dg.columns
        )
        dg["carrier"] = new_carriers
        return dg

    # Map carriers to REMIND technologies
    df = df.groupby(["year", "region", "carrier"], group_keys=False).apply(
        map_carriers_for_remind
    )

    return df


def weigh_by_REMIND_capacity(df):
    """
    Weighing here uses the capacities from REMIND and calaculates the weights
    s.t.

    the sum of weights equals 1 for each group of carriers (=
    "general_carrier") which are mapped against REMIND technologies.
    """
    # Read gen shares from REMIND for weighing
    capacity_weights = read_remind_data(
        file_path=snakemake.input["remind_weights"],
        variable_name="v32_shSeElDisp",
        rename_columns={
            "ttot": "year",
            "all_regi": "region",
            "all_te": "carrier",
        },
    )

    # Align dtypes & reduce to required columns
    capacity_weights = capacity_weights.astype(
        {"year": int, "level": float, "carrier": str, "region": str}
    )
    capacity_weights = capacity_weights[["year", "region", "carrier", "level"]]
    capacity_weights["level"] = capacity_weights["level"].where(
        lambda x: x > np.finfo(float).eps, 0.0
    )  # Remove very small values below EPS passed by GAMS

    # Map REMIND technologies to general_carriers
    capacity_weights["general_carrier"] = capacity_weights["carrier"].map(
        {lv: k for k, v in map_general_to_remind.items() for lv in v}
    )

    # Calculate weights for individual carrier based on total levels per shared "general_carrier" and individual share
    general_carrier_weights = capacity_weights.groupby(
        ["year", "region", "general_carrier"]
    )["level"].sum()
    general_carrier_weights = general_carrier_weights.where(
        lambda x: x != 0, 1
    )  # avoid division by zero
    general_carrier_weights.name = "general_carrier_weight"

    capacity_weights = capacity_weights.join(
        general_carrier_weights,
        on=["year", "region", "general_carrier"],
        validate="m:1",
    )
    capacity_weights["weight"] = (
        capacity_weights["level"] / capacity_weights["general_carrier_weight"]
    )

    # Map weights to data to-be-weighted based on (year, region, carrier)
    df = df.set_index(["year", "region", "carrier"]).join(
        capacity_weights.set_index(["year", "region", "carrier"])
    )

    # Consistency checks
    assert df["weight"].isna().sum() == 0, "Some weights are missing"
    assert all(df["weight"] >= 0.0), "Some weights are negative"
    assert all(df["weight"] <= 1.0), "Some weights are larger than 1.0"

    # Apply weights
    df["value"] *= df["weight"]

    return df[["value"]].reset_index()


# %%
# Real combining happens here
capacity_factors = postprocess_dataframe(capacity_factors)
generation_shares = postprocess_dataframe(generation_shares)
installed_capacities = postprocess_dataframe(installed_capacities)
market_values = postprocess_dataframe(market_values)
electricity_prices = postprocess_dataframe(electricity_prices)

# For loads only output AC (electricity) prices
electricity_prices = electricity_prices.query("carrier == 'AC'").drop(
    columns=["carrier"]
)
# %%
# Special treatment: Weigh values of df based on installed capacities in REMIND
generation_shares = weigh_by_REMIND_capacity(generation_shares)

if any(generation_shares.groupby(["year", "region"])["value"].sum() != 1.0):
    logger.warning(
        "Sum of generation shares is not equal to 1.0 for each year and region."
    )

# %%
# Export as csv values (informative purposes only, coupling parameters below via GDX)
for fn, df in {
    "capacity_factors": capacity_factors,
    "generation_shares": generation_shares,
    "installed_capacities": installed_capacities,
    "market_values": market_values,
    "electricity_prices": electricity_prices,
}.items():
    df.to_csv(snakemake.output[fn], index=False)

# %%
# Export to GAMS gdx file for coupling
gdx = gt.Container()

# Construct sets from exemplary index; luckily all data share most of the indeces
sets = {
    "year": market_values["year"].unique(),
    "region": market_values["region"].unique(),
    "carrier": market_values["carrier"].unique(),
}

# First add sets to the container
s_year = gt.Set(gdx, "year", records=sets["year"], description="simulation year")
s_region = gt.Set(
    gdx,
    "region",
    records=sets["region"],
    description="REMIND-EU region by which the values were aggregated",
)
s_carrier = gt.Set(
    gdx,
    "carrier",
    records=sets["carrier"],
    description="PyPSA technology by which the values were aggregated",
)

# Now we can add data to the container
c = gt.Parameter(
    gdx,
    name="capacity_factor",
    domain=[s_year, s_region, s_carrier],
    records=capacity_factors,
    description="Cacacity factors of technology per year and region in p.u.",
)

g = gt.Parameter(
    gdx,
    name="generation_share",
    domain=[s_year, s_region, s_carrier],
    records=generation_shares,
    description="Share of generation of technology per year and region in p.u.",
)

m = gt.Parameter(
    gdx,
    name="market_value",
    domain=[s_year, s_region, s_carrier],
    records=market_values,
    description="Market value of technology per year and region in EUR/MWh",
)

p = gt.Parameter(
    gdx,
    name="electricity_price",
    domain=[s_year, s_region],
    records=electricity_prices,
    description="Electricity price per year and region in EUR/MWh",
)

gdx.write(snakemake.output["gdx"])