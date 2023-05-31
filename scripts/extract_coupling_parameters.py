# -*- coding: utf-8 -*-
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa

def get_country_and_carrier(n, c):
    if "bus" not in n.df(c):
        bus = "bus0"  # for links
    else:
        bus = "bus"  # for all other components

    return [
        # Need to keep index alive, thus resetting before merge; get country information for PyPSA-EUR networks from each components associated bus
        n.df(c)
        .reset_index()
        .merge(n.df("Bus")[["country"]], left_on=bus, right_on="Bus")
        .set_index(c)["country"],
        # pypsa.statistics.get_carrier(n, c), # get nicer carrier names
        n.df(c).get("carrier"),  # get internal carrier names
    ]

def get_market_values(n):
    gtp = n.generators_t["p"]
    bmp = n.buses_t["marginal_price"]

    # Get buses associated with generators
    map_gen_to_bus = {
        gen: bus for gen, bus in zip(gtp.columns, gtp.columns.map(n.generators.bus))
    }

    # Calculate market values per generator
    total_market_value = (
        gtp.rename(columns=map_gen_to_bus) * bmp.loc[:, map_gen_to_bus.values()]
    ).sum()
    market_value = total_market_value / gtp.rename(columns=map_gen_to_bus).sum()

    # Restore generator names
    market_value.index = map_gen_to_bus.keys()

    # Add location (country) and carrier information
    market_value = market_value.to_frame(name="market_value")
    market_value = market_value.join(n.generators).merge(
        n.buses[["country"]], left_on="bus", right_on="Bus"
    )

    # Calculate market values per country & carrier, weighted by optimal capacity
    market_value = market_value.groupby(["country", "carrier"]).apply(
        lambda x: np.average(x["market_value"], weights=x["p_nom_opt"])
    )

    return market_value.to_frame("market_value")

if "snakemake" not in globals():
    # For testing only
    # TODO remove after testing
    from types import SimpleNamespace

    snakemake = SimpleNamespace()
    snakemake.input = {
        "networks": [
            "/mnt/c/Users/jhampp/Documents/GitHub/pik_hpc/coupling/pypsa-eur/results/no_scenario/networks/elec_s_4_y2130_i1_ec_lcopt_1H.nc",
            "/mnt/c/Users/jhampp/Documents/GitHub/pik_hpc/coupling/pypsa-eur/results/no_scenario/networks/elec_s_4_y2030_i1_ec_lcopt_1H.nc",
            "/mnt/c/Users/jhampp/Documents/GitHub/pik_hpc/coupling/pypsa-eur/results/no_scenario/networks/elec_s_4_y2100_i1_ec_lcopt_1H.nc",
        ]
    }
    snakemake.output = {
        "capacity_factors": "../results/no_scenario/coupling-parameters/i1/capacity_factors.csv",
        "market_values": "../results/no_scenario/coupling-parameters/i1/market_values.csv",
        "gdx": "../results/no_scenario/coupling-parameters/i1/coupling-parameters.gdx",
    }

cfs = []
mvs = []
for fp in snakemake.input["networks"]:
    # Extract year from filename, format: elec_y<YYYY>_<morestuff>.nc
    m = re.findall(r"elec_\S+_y([0-9]{4})_\S+\.nc", fp)
    assert len(m) == 1, "Unable to extract year from network path"
    year = m[0]

    # Load network
    network = pypsa.Network(fp)

    ## Extract coupling parameters from network

    # Capacity factors
    cf = network.statistics(comps=["Generator"], groupby=get_country_and_carrier)[
        ["Capacity Factor"]
    ]
    cf["year"] = year
    cf = cf.reset_index()  # index to columns
    cfs.append(cf)  # temporary hold Dataframes, to be combined later

    mv = get_market_values(network)
    mv["year"] = year
    mv = mv.reset_index()

    mvs.append(mv)
# %%
# Combine DataFrames and order / make nicer
cfs = pd.concat(cfs)
cfs = cfs.set_index(
    ["year", "country", "carrier"]
).sort_index()  # set index, more logical sort order
cfs = cfs.drop(columns=["level_0"])  # not needed

mvs = pd.concat(mvs)
mvs = mvs.set_index(
    ["year", "country", "carrier"]
).sort_index()  # set index, more logical sort order
# %%
cfs.to_csv(snakemake.output["capacity_factors"])
mvs.to_csv(snakemake.output["market_values"])

from gams import transfer as gt

gdx = gt.Container()

sets = {i: mvs.index.get_level_values(i).unique() for i in mvs.index.names}

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
m = gt.Parameter(
    gdx,
    name="market_value",
    domain=[s_year, s_country, s_carrier],
    records=mvs.reset_index(),
    description="Market value of technology in year and country in EUR/MWh",
)

c = gt.Parameter(
    gdx,
    name="capacity_factor",
    domain=[s_year, s_country, s_carrier],
    records=mvs.reset_index(),
    description="Cacacity factors of technology in year and country in p.u.",
)

gdx.write(snakemake.output["gdx"])

# TODO: Dis-aggregation PyPSA2REMIND