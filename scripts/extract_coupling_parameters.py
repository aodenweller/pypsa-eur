# -*- coding: utf-8 -*-
import re
from pathlib import Path

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
        pypsa.statistics.get_carrier(n, c),
    ]


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
    "capacity_factors": "../results/no_scenario/coupling-parameters/i1/capacity_factors.csv"
}

cfs = []
for fp in snakemake.input["networks"]:
    # Extract year from filename
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
    cf = cf.reset_index() # index to columns
    cfs.append(cf) # temporary hold Dataframes, to be combined later

# Combine DataFrames and order / make nicer
cfs = pd.concat(cfs)
cfs = cfs.set_index(["year", "country", "carrier"]).sort_index() # set index, more logical sort order
cfs = cfs.drop(columns=["level_0"]) # not needed

cfs.to_csv(snakemake.output["capacity_factors"])
