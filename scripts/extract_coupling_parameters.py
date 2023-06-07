# -*- coding: utf-8 -*-
# %%
import re
from pathlib import Path
from gams import transfer as gt

import numpy as np
import pandas as pd
import pypsa

# %%
# Use a two step mapping approach between PyPSA-EUR and REMIND:
# First mapping is aggregating PyPSA-EUR technologies to general technologies
# Second mapping is disaggregating general technologies to REMIND technologies
map_pypsaeur_to_general = {
    "CCGT": "CCGT",
    "OCGT": "OCGT",
    "biomass": "biomass",
    "coal": "all_coal",
    "lignite": "all_coal",
    "offwind-ac": "wind_offshore",
    "offwind-dc": "wind_offshore",
    "oil": "oil",
    "onwind": "wind_onshore",
    "solar": "solar_pv",
    "nuclear": "nuclear",
}

map_general_to_remind = {
    "CCGT": "ngcc",
    "CCGT": "ngccc",
    "CCGT": "gaschp",
    "OCGT": "ngt",
    "biomass": "biochp",
    "biomass": "bioigcc",
    "biomass": "bioigccc",
    "all_coal": "igcc",
    "all_coal": "igccc",
    "all_coal": "pc",
    "all_coal": "coalchp",
    "nuclear": "tnrs",
    "nuclear": "fnrs",
    "oil": "dot",
    "solar_pv": "spv",
    "wind_offshore": "windoff",
    "wind_onshore": "wind",
}

# %%
def check_for_missing_carriers(n):
    tmp_set = set(n.generators['carrier']) - map_pypsaeur_to_general.keys()
    if tmp_set:
        print(
            f"The following technologies (carriers) are missing in the mapping PyPSA-EUR -> general technologies: "
            f"{tmp_set}"
        )

    tmp_set = map_pypsaeur_to_general.values() - map_general_to_remind.keys()
    if tmp_set:
        print(
            f"The following technologies are missing in the mapping general technologies -> REMIND: "
            f"{tmp_set}"
        )


# %%
if "snakemake" not in globals():
    # For testing only
    # TODO remove after testing
    from types import SimpleNamespace

    snakemake = SimpleNamespace()
    snakemake.input = {
        "networks": [
            "../results/no_scenario/networks/elec_s_4_y2025_i1_ec_lcopt_1H.nc",
            "../results/no_scenario/networks/elec_s_4_y2030_i1_ec_lcopt_1H.nc",
            "../results/no_scenario/networks/elec_s_4_y2035_i1_ec_lcopt_1H.nc",
            "../results/no_scenario/networks/elec_s_4_y2040_i1_ec_lcopt_1H.nc",
            "../results/no_scenario/networks/elec_s_4_y2045_i1_ec_lcopt_1H.nc",
            "../results/no_scenario/networks/elec_s_4_y2050_i1_ec_lcopt_1H.nc",
            "../results/no_scenario/networks/elec_s_4_y2055_i1_ec_lcopt_1H.nc",
            "../results/no_scenario/networks/elec_s_4_y2060_i1_ec_lcopt_1H.nc",
            "../results/no_scenario/networks/elec_s_4_y2070_i1_ec_lcopt_1H.nc",
            "../results/no_scenario/networks/elec_s_4_y2080_i1_ec_lcopt_1H.nc",
            "../results/no_scenario/networks/elec_s_4_y2090_i1_ec_lcopt_1H.nc",
            "../results/no_scenario/networks/elec_s_4_y2100_i1_ec_lcopt_1H.nc",
            "../results/no_scenario/networks/elec_s_4_y2110_i1_ec_lcopt_1H.nc",
            "../results/no_scenario/networks/elec_s_4_y2130_i1_ec_lcopt_1H.nc",
            "../results/no_scenario/networks/elec_s_4_y2150_i1_ec_lcopt_1H.nc",
        ]
    }
    snakemake.output = {
        "capacity_factors": "../results/no_scenario/coupling-parameters/i1/capacity_factors.csv",
        "generation_shares": "../results/no_scenario/coupling-parameters/i1/generation_shares.csv",
        "installed_capacities": "../results/no_scenario/coupling-parameters/i1/installed_capacities.csv",
        "market_values": "../results/no_scenario/coupling-parameters/i1/market_values.csv",
        "electricity_prices": "../results/no_scenario/coupling-parameters/i1/electricity_prices.csv",
        "gdx": "../results/no_scenario/coupling-parameters/i1/coupling-parameters.gdx",
    }

# %%
capacity_factors = []
generation_shares = []
installed_capacities = []
market_values = []
electricity_prices = []

for fp in snakemake.input["networks"]:
    # Extract year from filename, format: elec_y<YYYY>_<morestuff>.nc
    m = re.findall(r"elec_\S+_y([0-9]{4})_\S+\.nc", fp)
    assert len(m) == 1, "Unable to extract year from network path"
    year = m[0]

    # Load network
    network = pypsa.Network(fp)
    check_for_missing_carriers(network)

    
    # Add information for aggregation later: country name and general technology
    network.generators["country"] = network.generators["bus"].map(network.buses["country"])
    network.generators["general_carrier"] = network.generators["carrier"].map(
        map_pypsaeur_to_general
    )
    network.loads["country"] = network.loads["bus"].map(network.buses["country"])
    network.loads["bus_carrier"] = network.loads["bus"].map(network.buses["carrier"])
    
    ## Extract coupling parameters from network
    # Calculate capacity factors; by assigning the general carrier and grouping by here, the capacity factor is automatically
    # calculated across all "carrier" technologies that map to the same "general_carrier"
    capacity_factor = network.statistics(comps=["Generator"], groupby=["country", "general_carrier"])[
        "Capacity Factor"
    ]
    capacity_factor = capacity_factor.to_frame("value").reset_index()
    capacity_factor["year"] = year
    capacity_factors.append(capacity_factor)

    # Calculate shares of technologies in annual generation
    generation_share = (
        network.statistics(
            comps=["Generator"],
            groupby=["country", "general_carrier"],
            aggregate_time="sum",
        )["Supply"]
        / network.statistics(
            comps=["Generator"],
            groupby=["country", "general_carrier"],
            aggregate_time="sum",
        )["Supply"].sum()
    )
    generation_share = generation_share.to_frame("value").reset_index()
    generation_share["year"] = year
    generation_shares.append(generation_share)
    
    # Calculate technology installed capacities
    installed_capacity = network.statistics(comps=["Generator"], groupby=["country", "general_carrier"])["Optimal Capacity"]
    installed_capacity = installed_capacity.to_frame("value").reset_index()
    installed_capacity["year"] = year
    installed_capacities.append(installed_capacity)

    # Calculate the market values (round-about way as the intended method of the statistics module is not yet available)
    market_value = (
        network.statistics(comps=["Generator"], groupby=["country", "general_carrier"], aggregate_time="sum")["Revenue"]
        /
        network.statistics(comps=["Generator"], groupby=["country", "general_carrier"], aggregate_time="sum")["Supply"]
    )
    market_value = market_value.to_frame("value").reset_index()
    market_value["year"] = year
    market_values.append(market_value)
    
    # Calculate load-weighted electricity prices based on bus marginal prices
    electricity_price = ( 
    network.statistics(comps=["Load"], groupby=["country","bus_carrier"], aggregate_time="sum")["Revenue"]
    /
    (-1 * network.statistics(comps=["Load"], groupby=["country","bus_carrier"], aggregate_time="sum")["Withdrawal"])
    )
    electricity_price = electricity_price.to_frame("value").reset_index()
    electricity_price["year"] = year
    electricity_prices.append(electricity_price)

# %%
## Combine DataFrames to same format
# Helper function
def postprocess_dataframe(df):
    """General function to postprocess the dataframes, combines the network-specific results into one dataframe
    removes excess columns / sets index and sorts by country + year"""
    df = pd.concat(df)
    df = df.rename(columns={"general_carrier": "carrier", "bus_carrier": "carrier"}) # different auxiliary columns have different names; rename for consistency
    df = df.set_index(["year", "country", "carrier"]).sort_index()  # set index, more logical sort order
    return df[["value"]]

# Real combining happens here
capacity_factors = postprocess_dataframe(capacity_factors)
generation_shares = postprocess_dataframe(generation_shares)
installed_capacities = postprocess_dataframe(installed_capacities)
market_values = postprocess_dataframe(market_values)
electricity_prices = postprocess_dataframe(electricity_prices)

# %%
# Export as csv values (informative purposes only, coupling parameters below via GDX)
capacity_factors.to_csv(snakemake.output["capacity_factors"])
generation_shares.to_csv(snakemake.output["generation_shares"])
installed_capacities.to_csv(snakemake.output["installed_capacities"])
market_values.to_csv(snakemake.output["market_values"])
electricity_prices.to_csv(snakemake.output["electricity_prices"])

# %%
# Export to GAMS gdx file for coupling
gdx = gt.Container()

# Construct sets from exemplary index; luckily all data shares the same indeces
sets = {i: market_values.index.get_level_values(i).unique() for i in market_values.index.names}

# First add sets to the container
s_year = gt.Set(gdx, "year", records=sets["year"], description="simulation year")
s_country = gt.Set(
    gdx,
    "country",
    records=sets["country"],
    description="country by which the values were aggregated",
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
    domain=[s_year, s_country, s_carrier],
    records=capacity_factors.reset_index(),
    description="Cacacity factors of technology per year and country in p.u.",
)

g = gt.Parameter(
    gdx,
    name="generation_share",
    domain=[s_year, s_country, s_carrier],
    records=generation_shares.reset_index(),
    description="Share of generation of technology per year and country in p.u.",
)

i = gt.Parameter(
    gdx,
    name="installed_capacity",
    domain=[s_year, s_country, s_carrier],
    records=installed_capacities.reset_index(),
    description="Installed capacity of technology per year and country in MW",
)

m = gt.Parameter(
    gdx,
    name="market_value",
    domain=[s_year, s_country, s_carrier],
    records=market_values.reset_index(),
    description="Market value of technology per year and country in EUR/MWh",
)

p = gt.Parameter(
    gdx,
    name="electricity_price",
    domain=[s_year, s_country],
    records=electricity_prices.loc[:,:,"AC"].reset_index(), # only electricity prices, remaining loads (if any) are ignored
    description="Electricity price per year and country in EUR/MWh",
)

gdx.write(snakemake.output["gdx"])

# TODO: Dis-aggregation PyPSA2REMIND
