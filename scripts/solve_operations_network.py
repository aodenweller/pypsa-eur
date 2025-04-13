# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2017-2024 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
"""
Solves linear optimal dispatch in hourly resolution using the capacities of
previous capacity expansion in rule :mod:`solve_network`.
"""
#%%

import logging
import time

import numpy as np
import pypsa
from _helpers import (
    configure_logging,
    set_scenario_config,
    update_config_from_wildcards,
    setup_gurobi_tunnel_and_env,
    check_gurobi_license,
)
from solve_network import prepare_network, solve_network

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "solve_operations_network",
            scenario="PyPSA_PkBudg1000_DEU_allRCL_PyPSArefactor_perturb_4nodes_2025-04-10_15.50.49",
            iteration="1",
            year="2035",
            simpl="",
            opts="3H-Ep133.1",
            clusters="4",
            ll="copt",
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)
    update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    solve_opts = snakemake.params.options

    np.random.seed(solve_opts.get("seed", 123))
#%%

    # Create empty output trigger file to track that this rule ran once
    with open(snakemake.output.trigger, "w") as f:
        f.write("")
        
    # Load network
    n = pypsa.Network(snakemake.input.network)
    
    # Exit if if there is no objective
    if not hasattr(n, "objective"):
        logger.warning("Network has no objective. Cannot solve operations network.")
        exit(0)

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

    #n.optimize.fix_optimal_capacities()
    
    # Multiply all p_nom of generators with 1 + 1E-4 to avoid numerical issues with Gurobi
    tolerance = snakemake.params.get("remind_settings")["tolerance"]
    tolerance = float(tolerance)
    
    logger.info(f"Multiplying all capacities by 1+{tolerance}")
    
    for c, attr in pypsa.descriptors.nominal_attrs.items():
        ext_i = n.get_extendable_i(c)
        n.static(c).loc[ext_i, attr] = (1 + tolerance) * n.static(c).loc[ext_i, attr + "_opt"]
        n.static(c)[attr + "_extendable"] = False
        

    n = prepare_network(n, solve_opts, config=snakemake.config)
    n = solve_network(
        n,
        config=snakemake.config,
        params=snakemake.params,
        solving=snakemake.params.solving,
        log_fn=snakemake.log.solver,
    )

    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    
    # Use snakemake.input.network and replace _trigger by .nc
    filename = snakemake.output["trigger"].replace("_trigger", ".nc")
    
    n.export_to_netcdf(filename)
    
    time.sleep(5)
