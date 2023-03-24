# SPDX-FileCopyrightText: : 2023 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
#
# File for REMIND coupling by Adrian Odenweller (adrian.odenweller@pik-potsdam.de)

# This file contains all rules from add_electricity onwards, all with a "remind" suffix 
# All rules have an additional wildcard "year" to facilitate parallelisation of years
# Some rules also have an additional wildcard "iter" to keep track of REMIND iterations

rule add_electricity_remind:
    input:
        **{
            f"profile_{tech}": RESOURCES + f"profile_{tech}.nc"
            for tech in config["electricity"]["renewable_carriers"]
        },
        **{
            f"conventional_{carrier}_{attr}": fn
            for carrier, d in config.get("conventional", {None: {}}).items()
            for attr, fn in d.items()
            if str(fn).startswith("data/")
        },
        base_network=RESOURCES + "networks/base.nc",
        tech_costs=COSTS,
        tech_costs_remind=RESOURCES_remind + "costs_y{year}.csv",
        regions=RESOURCES + "regions_onshore.geojson",
        powerplants=RESOURCES_remind + "powerplants_y{year}.csv",
        hydro_capacities=ancient("data/bundle/hydro_capacities.csv"),
        geth_hydro_capacities="data/geth2015_hydro_capacities.csv",
        load=RESOURCES_remind + "load_y{year}.csv",
        nuts3_shapes=RESOURCES + "nuts3_shapes.geojson",
    output:
        RESOURCES_remind + "networks/elec_y{year}.nc",
    log:
        LOGS_remind + "add_electricity_y{year}.log",
    benchmark:
        BENCHMARKS_remind + "add_electricity_y{year}"
    threads: 1
    resources:
        mem_mb=5000,
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/add_electricity.py"


rule simplify_network_remind:
    input:
        network=RESOURCES_remind + "networks/elec_y{year}.nc",
        tech_costs=COSTS,
        regions_onshore=RESOURCES + "regions_onshore.geojson",
        regions_offshore=RESOURCES + "regions_offshore.geojson",
    output:
        network=RESOURCES_remind + "networks/elec_s{simpl}_y{year}.nc",
        regions_onshore=RESOURCES_remind + "regions_onshore_elec_s{simpl}_y{year}.geojson",
        regions_offshore=RESOURCES_remind + "regions_offshore_elec_s{simpl}_y{year}.geojson",
        busmap=RESOURCES_remind + "busmap_elec_s{simpl}_y{year}.csv",
        connection_costs=RESOURCES_remind + "connection_costs_s{simpl}_y{year}.csv",
    log:
        LOGS_remind + "simplify_network/elec_s{simpl}_y{year}.log",
    benchmark:
        BENCHMARKS_remind + "simplify_network/elec_s{simpl}_y{year}"
    threads: 1
    resources:
        mem_mb=4000,
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/simplify_network.py"


rule cluster_network_remind:
    input:
        network=RESOURCES_remind + "networks/elec_s{simpl}_y{year}.nc",
        regions_onshore=RESOURCES_remind + "regions_onshore_elec_s{simpl}_y{year}.geojson",
        regions_offshore=RESOURCES_remind + "regions_offshore_elec_s{simpl}_y{year}.geojson",
        busmap=ancient(RESOURCES_remind + "busmap_elec_s{simpl}_y{year}.csv"),
        custom_busmap=(
            "data/custom_busmap_elec_s{simpl}_{clusters}.csv"
            if config["enable"].get("custom_busmap", False)
            else []
        ),
        tech_costs=COSTS,
    output:
        network=RESOURCES_remind + "networks/elec_s{simpl}_{clusters}_y{year}.nc",
        regions_onshore=RESOURCES_remind + "regions_onshore_elec_s{simpl}_{clusters}_y{year}.geojson",
        regions_offshore=RESOURCES_remind + "regions_offshore_elec_s{simpl}_{clusters}_y{year}.geojson",
        busmap=RESOURCES_remind + "busmap_elec_s{simpl}_{clusters}_y{year}.csv",
        linemap=RESOURCES_remind + "linemap_elec_s{simpl}_{clusters}_y{year}.csv",
    log:
        LOGS_remind + "cluster_network/elec_s{simpl}_{clusters}_y{year}.log",
    benchmark:
        BENCHMARKS_remind + "cluster_network/elec_s{simpl}_{clusters}_y{year}"
    threads: 1
    resources:
        mem_mb=6000,
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/cluster_network.py"


rule add_extra_components_remind:
    input:
        network=RESOURCES_remind + "networks/elec_s{simpl}_{clusters}_y{year}.nc",
        tech_costs=COSTS,
    output:
        network=RESOURCES_remind + "networks/elec_s{simpl}_{clusters}_y{year}_ec.nc",
    log:
        LOGS_remind + "add_extra_components/elec_s{simpl}_{clusters}_y{year}.log",
    benchmark:
        BENCHMARKS_remind + "add_extra_components/elec_s{simpl}_{clusters}_y{year}_ec"
    threads: 1
    resources:
        mem_mb=3000,
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/add_extra_components.py"


rule prepare_network_remind:
    input:
        network=RESOURCES_remind + "networks/elec_s{simpl}_{clusters}_y{year}_ec.nc",
        tech_costs=COSTS,
    output:
        network=RESOURCES_remind + "networks/elec_s{simpl}_{clusters}_y{year}_ec_l{ll}_{opts}.nc",
    log:
        LOGS_remind + "prepare_network/elec_s{simpl}_{clusters}_y{year}_ec_l{ll}_{opts}.log",
    benchmark:
        (BENCHMARKS_remind + "prepare_network/elec_s{simpl}_{clusters}_y{year}_ec_l{ll}_{opts}")
    threads: 1
    resources:
        mem_mb=4000,
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/prepare_network.py"

# Introduce wildcard {iter} from here to distinguish iterations in the result files
# {iter} gets passed on from REMIND via a command line parameter
# TODO: Check if {iter} only from here makes sense?
rule solve_network_remind:
    input:
        network=RESOURCES_remind + "networks/elec_s{simpl}_{clusters}_y{year}_ec_l{ll}_{opts}.nc",
    output:
        network=RESULTS_remind + "networks/elec_s{simpl}_{clusters}_y{year}_i{iter}_ec_l{ll}_{opts}.nc",
    log:
        solver=normpath(
            LOGS_remind + "solve_network/elec_s{simpl}_{clusters}_y{year}_i{iter}_ec_l{ll}_{opts}_solver.log"
        ),
        python=LOGS_remind
        + "solve_network/elec_s{simpl}_{clusters}_y{year}_i{iter}_ec_l{ll}_{opts}_python.log",
        memory=LOGS_remind
        + "solve_network/elec_s{simpl}_{clusters}_y{year}_i{iter}_ec_l{ll}_{opts}_memory.log",
    benchmark:
        BENCHMARKS_remind + "solve_network/elec_s{simpl}_{clusters}_y{year}_i{iter}_ec_l{ll}_{opts}"
    threads: 4
    resources:
        mem_mb=memory,
    shadow:
        "minimal"
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/solve_network.py"

# Additional rule that solves all networks
rule solve_all_networks_remind:
    input:
        expand(
            RESULTS_remind + "networks/elec_s{simpl}_{clusters}_y{year}_i{iter}_ec_l{ll}_{opts}.nc",
            **config["scenario"],
            **config  # Necessary for "iter" wildcard
        )

# Additional rule that exports the network to csv files
rule export_network_remind:
    input:
        RESULTS_remind + "elec_s{simpl}_{clusters}_y{year}_i{iter}_ec_l{ll}_{opts}.nc",
    output:
        directory(RESULTS_remind + "out_elec_s{simpl}_{clusters}_y{year}_i{iter}_ec_l{ll}_{opts}"),
    log:
        LOGS_remind + "export_network/elec_s{simpl}_{clusters}_y{year}_i{iter}_ec_l{ll}_{opts}.log",
    run:
        import pypsa
        n = pypsa.Network(input[0])
        n.export_to_csv_folder(output[0])

# Additional rule that exports all networks to csv files
rule export_all_networks_remind:
    input:
        expand(
            RESULTS_remind + "out_elec_s{simpl}_{clusters}_y{year}_i{iter}_ec_l{ll}_{opts}",
            **config["scenario"],
            **config  # Neceswsary for "iter" wildcard
        )