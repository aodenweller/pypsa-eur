# SPDX-FileCopyrightText: : 2023-2024 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT


rule solve_network:
    params:
        solving=config_provider("solving"),
        foresight=config_provider("foresight"),
        planning_horizons=config_provider("scenario", "planning_horizons"),
        co2_sequestration_potential=config_provider(
            "sector", "co2_sequestration_potential", default=200
        ),
        custom_extra_functionality=input_custom_extra_functionality,
    input:
        network=ITERATION_RESOURCES
        + "y{year}/networks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}.nc",
        RCL_p_nom_limits=ITERATION_RESOURCES + "y{year}/RCL_p_nom_limits.csv",
        region_mapping="config/regionmapping_21_EU11.csv",
        technology_cost_mapping="config/technology_cost_mapping.csv",
    output:
        network=ITERATION_RESULTS
        + "y{year}/networks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}.nc",
        config=ITERATION_RESULTS + "y{year}/configs/config.elec_s{simpl}_{clusters}_ec_l{ll}_{opts}.yaml",
    log:
        solver=normpath(
            ITERATION_LOGS
            + "y{year}/solve_network/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_solver.log"
        ),
        python=ITERATION_LOGS
        + "y{year}/solve_network/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_python.log",
    benchmark:
        (
            ITERATION_BENCHMARKS
            + "y{year}/solve_network/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}"
        )
    threads: solver_threads
    group:
        "iy"
    resources:
        mem_mb=memory,
        runtime=config_provider("solving", "runtime", default="6h"),
    shadow:
        "shallow"
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/solve_network.py"


rule solve_operations_network:
    params:
        options=config_provider("solving", "options"),
        solving=config_provider("solving"),
        foresight=config_provider("foresight"),
        planning_horizons=config_provider("scenario", "planning_horizons"),
        co2_sequestration_potential=config_provider(
            "sector", "co2_sequestration_potential", default=200
        ),
        custom_extra_functionality=input_custom_extra_functionality,
    input:
        network=RESULTS + "networks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}.nc",
    output:
        network=RESULTS + "networks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_op.nc",
    log:
        solver=normpath(
            RESULTS
            + "logs/solve_operations_network/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_op_solver.log"
        ),
        python=RESULTS
        + "logs/solve_operations_network/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_op_python.log",
    benchmark:
        (
            RESULTS
            + "benchmarks/solve_operations_network/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}"
        )
    threads: 4
    resources:
        mem_mb=(lambda w: 10000 + 372 * int(w.clusters)),
        runtime=config_provider("solving", "runtime", default="6h"),
    shadow:
        "shallow"
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/solve_operations_network.py"
