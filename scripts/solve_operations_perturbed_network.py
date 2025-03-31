# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2017-2024 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
#%%
"""
Solves linear optimal dispatch in hourly resolution using the capacities of
previous capacity expansion optimization, and perturbing the 
"""

import logging

import numpy as np
import pypsa
from _helpers import (
    configure_logging,
    set_scenario_config,
    update_config_from_wildcards,
    get_technology_mapping,
    setup_gurobi_tunnel_and_env,
    check_gurobi_license,
)
from solve_network import prepare_network, solve_network

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "solve_operations_perturbed_network",
            scenario="TESTsmk",
            iteration="6",
            year="2100",
            simpl="",
            opts="3H-Ep366.3",
            clusters="4",
            ll="copt",
            ptech="coal",
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)
    update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    solve_opts = snakemake.params.solving["options"]

    np.random.seed(solve_opts.get("seed", 123))

#%%
    n = pypsa.Network(snakemake.input.network)

    n.optimize.fix_optimal_capacities()
    
    # Read technology mapping
    tech_mapping = get_technology_mapping(
        snakemake.input["technology_cost_mapping"], group_technologies=True
    )
    
    # Perturb the network for technoloy ptech in wildcards
    ptech = snakemake.wildcards.ptech
    
    # Find the corresponding technology in tech_mapping
    # ptech should always correspond to the first word in technology_group
    ptech_pypsa = tech_mapping.loc[
        tech_mapping["technology_group"].str.contains(ptech),
        "PyPSA-Eur"].unique()
    
    # Perturbation factor from config
    perturbation_factor = snakemake.params["perturbation"]["perturbation_factor"]
    min_perturbation = snakemake.params["perturbation"]["min_perturbation"]
    
    optimal_capacity = n.generators.loc[n.generators.carrier.isin(ptech_pypsa), "p_nom_opt"].sum()
    perturbed_capacity = optimal_capacity * perturbation_factor
    
    if perturbed_capacity - optimal_capacity > min_perturbation:
        n.generators.loc[n.generators.carrier.isin(ptech_pypsa), "p_nom"] *= perturbation_factor
    else:
        n.generators.loc[n.generators.carrier.isin(ptech_pypsa), "p_nom"] *= (1 + min_perturbation / optimal_capacity)
        
    # Create empty output trigger file to track that this rule ran once
    with open(snakemake.output.trigger, "w") as f:
        f.write("")
    
    # deal with the gurobi license activation, which requires a tunnel to the login nodes
    solver_config = snakemake.config["solving"]["solver"]
    gurobi_license_config = snakemake.config["solving"].get("gurobi_hpc_tunnel", None)
    logger.info(f"Solver config {solver_config} and license cfg {gurobi_license_config}")
    if (solver_config["name"] == "gurobi") & (gurobi_license_config is not None):
        setup_gurobi_tunnel_and_env(gurobi_license_config, logger=logger)
        
    # Check that gurobi license is available
    # Note: Although linopy supports passing the Gurobi environment via env
    # this doesn't work here as we need to create the Gurobi environment
    # in a subprocess in order to catch possible timeouts.
    # Gurobi environments cannot be shared across resources.
    check_gurobi_license()

    n = prepare_network(n, solve_opts, config=snakemake.config)
    n = solve_network(
        n,
        config=snakemake.config,
        params=snakemake.params,
        solving=snakemake.params.solving,
        log_fn=snakemake.log.solver,
    )

    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    n.export_to_netcdf(snakemake.output[0])

# %%
