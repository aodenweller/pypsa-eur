# -*- coding: utf-8 -*-
# This script determines the wildcards for the perturbed scenarios

# %%
import logging
import numpy as np
import pandas as pd
import pypsa
import re
from export_to_REMIND import (
    get_pypsa_to_remind_region_mapping,
    add_columns_for_processing,
    get_pypsa_to_general_mapping,
)

from _helpers import (
    configure_logging,
    mock_snakemake,
    get_region_mapping,
)

logger = logging.getLogger(__name__)

# %%

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "determine_perturbed_op_networks",
            configfiles="config/config.remind.yaml",
            iteration="9",
            scenario="PyPSA_PkBudg1000_DEU_allRCL_PyPSArefactor_4nodes_Loadshedding_2025-04-11_14.02.50",
        )
        fp_networks = [
            f"../results/{snakemake.wildcards['scenario']}/i{snakemake.wildcards['iteration']}/y2030/networks/elec_s_4_ec_lcopt_3H-Ep131.8.nc",
            f"../results/{snakemake.wildcards['scenario']}/i{snakemake.wildcards['iteration']}/y2035/networks/elec_s_4_ec_lcopt_3H-Ep133.1.nc",
            f"../results/{snakemake.wildcards['scenario']}/i{snakemake.wildcards['iteration']}/y2080/networks/elec_s_4_ec_lcopt_3H-Ep299.9.nc",
        ]
    else:
        fp_networks = snakemake.input["networks"]

    configure_logging(snakemake)

    # Read the config file
    remind_settings = snakemake.params.get("remind_settings")
    
    if remind_settings["enable"]:
        # Read original scenario wildcards csv
        df = pd.read_csv(snakemake.input["scenario_wildcards"])
        # Initialise
        df["ptech"] = pd.Series([None] * len(df), dtype=object)
        # Add ptech column conditionally on switch in REMIND
        ptech = remind_settings["generators"]
        # Get region mapping
        region_mapping = get_pypsa_to_remind_region_mapping(
            snakemake.input["region_mapping"]
        )
        # Get PyPSA-EUR to general technology mapping
        map_pypsaeur_to_general = get_pypsa_to_general_mapping(
            snakemake.input["technology_cost_mapping"]
        )
        # Loop over all networks
        for fp in fp_networks:
            # Read the network
            n = pypsa.Network(fp)
            year = int(re.search(r"y(\d{4})", fp).group(1))
            if not hasattr(n, "objective"):
                logger.warning(f"Network {fp} does not have an objective. Skipping.")
                # Remove year
                idx = df.index[df["year"] == year]
                df.drop(idx, inplace=True)
                continue
            # Add region and general_carrier columns
            add_columns_for_processing(n, region_mapping, map_pypsaeur_to_general)
            # Get year
            # Calculate optimal capacities
            opt_cap = n.statistics.optimal_capacity(
                comps=["Generator"], groupby=["region", "general_carrier"], nice_names=False
            ).reset_index()
            # Calculate share of capacity
            opt_cap["share"] = opt_cap["p_nom_opt"] / opt_cap.groupby("region")[
                "p_nom_opt"
            ].transform("sum")
            # Only include if share is greater than remind_settings["min_capacity_share"]
            opt_cap = opt_cap[opt_cap["share"] > remind_settings["min_capacity_share"]]
            # Add ptech column and try to match partially the general_carrier
            opt_cap["ptech"] = opt_cap["general_carrier"].apply(
                lambda x: [p for p in ptech if p in str(x)]
            )
            # Make from list to string
            opt_cap["ptech"] = opt_cap["ptech"].apply(lambda x: ",".join(x))
            # Remove if ptech is empty
            opt_cap = opt_cap[opt_cap["ptech"] != ""]
            # Add to the corresponding year of df
            idx = df.index[df["year"] == year]
            df.at[idx[0], "ptech"] = list(opt_cap["ptech"].unique())
        # Explode with ptech
        df = df.explode("ptech", ignore_index=True)
    else:
        df = pd.DataFrame()

    # Write to output
    df.to_csv(snakemake.output["scenario_wildcards_perturbed"], index=False)
