# SPDX-FileCopyrightText: : 2023 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT


rule solve_network:
    params:
        solving=config["solving"],
        foresight=config["foresight"],
        planning_horizons=config["scenario"]["planning_horizons"],
        co2_sequestration_potential=config["sector"].get(
            "co2_sequestration_potential", 200
        ),
        custom_extra_functionality=input_custom_extra_functionality,
    input:
        network=ITERATION_RESOURCES
        + "y{year}/networks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}.nc",
        RCL_p_nom_limits=ITERATION_RESOURCES + "y{year}/RCL_p_nom_limits.csv",
        region_mapping="config/regionmapping_21_EU11.csv",
        technology_cost_mapping="config/technology_cost_mapping.csv",
        config=ITERATION_RESULTS + "config.yaml",
    output:
        network=ITERATION_RESULTS
        + "y{year}/networks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}.nc",
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
    threads: 4
    group:
        "iy"
    resources:
        mem_mb=memory,
        walltime=config["solving"].get("walltime", "12:00:00"),
    shadow:
        "minimal"
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/solve_network.py"


rule solve_operations_network:
    params:
        options=config["solving"]["options"],
    input:
        network=RESULTS + "networks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}.nc",
    output:
        network=RESULTS + "networks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_op.nc",
    log:
        solver=normpath(
            LOGS
            + "solve_operations_network/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_op_solver.log"
        ),
        python=LOGS
        + "solve_operations_network/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_op_python.log",
    benchmark:
        (
            BENCHMARKS
            + "solve_operations_network/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}"
        )
    threads: 4
    resources:
        mem_mb=(lambda w: 10000 + 372 * int(w.clusters)),
        walltime=config["solving"].get("walltime", "12:00:00"),
    shadow:
        "minimal"
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/solve_operations_network.py"
