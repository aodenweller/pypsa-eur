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
    input:
        network="resources/{scenario}/i{iteration}/y{year}/networks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}.nc",
        RCL_p_nom_limits="resources/{scenario}/i{iteration}/y{year}/RCL_p_nom_limits.csv",
        region_mapping="config/regionmapping_21_EU11.csv",
        technology_mapping="config/technology_mapping.csv",
    output:
        network="results/{scenario}/i{iteration}/y{year}/networks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}.nc",
    log:
        solver=normpath(
            LOGS + "{scenario}/i{iteration}/y{year}/solve_network/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_solver.log"
        ),
        python=LOGS + "{scenario}/i{iteration}/y{year}/solve_network/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_python.log",
    benchmark:
        BENCHMARKS + "{scenario}/i{iteration}/y{year}/solve_network/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}"
    threads: 4
    group:
        "iy"
    resources:
        mem_mb=memory,
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
        mem_mb=(lambda w: 5000 + 372 * int(w.clusters)),
    shadow:
        "minimal"
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/solve_operations_network.py"
