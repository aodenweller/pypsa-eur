# SPDX-FileCopyrightText: : 2023-2024 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
# This file contains those rules of build_sector.smk that are required for the REMIND coupling
# This currently encompasses anything related to electricity demand for EVs
# In the future this might be extended to electricity demand for heating (and potentially others)


# Unchanged
rule build_population_layouts:
    input:
        nuts3_shapes=resources("nuts3_shapes.geojson"),
        urban_percent="data/worldbank/API_SP.URB.TOTL.IN.ZS_DS2_en_csv_v2.csv",
        cutout=lambda w: "cutouts/"
        + CDIR
        + config_provider("atlite", "default_cutout")(w)
        + ".nc",
    output:
        pop_layout_total=resources("pop_layout_total.nc"),
        pop_layout_urban=resources("pop_layout_urban.nc"),
        pop_layout_rural=resources("pop_layout_rural.nc"),
    log:
        logs("build_population_layouts.log"),
    resources:
        mem_mb=20000,
    benchmark:
        benchmarks("build_population_layouts")
    threads: 8
    group:
        "iy"
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/build_population_layouts.py"

# Paths have been modified, but script is unchanged
rule build_clustered_population_layouts:
    input:
        pop_layout_total=resources("pop_layout_total.nc"),
        pop_layout_urban=resources("pop_layout_urban.nc"),
        pop_layout_rural=resources("pop_layout_rural.nc"),
        regions_onshore=SCENARIO_RESOURCES
        + "i{iteration}/y{year}/regions_onshore_elec_s{simpl}_{clusters}.geojson",
        cutout=lambda w: "cutouts/"
        + CDIR
        + config_provider("atlite", "default_cutout")(w)
        + ".nc",
    output:
        clustered_pop_layout=SCENARIO_RESOURCES
        + "i{iteration}/y{year}/pop_layout_elec_s{simpl}_{clusters}.csv",
    log:
        SCENARIO_LOGS
        + "i{iteration}/y{year}/build_clustered_population_layouts_{simpl}_{clusters}.log",
    resources:
        mem_mb=10000,
    benchmark:
        SCENARIO_BENCHMARKS
        + "i{iteration}/y{year}/build_clustered_population_layouts/s{simpl}_{clusters}"
    group:
        "iy"
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/build_clustered_population_layouts.py"

# Unchanged
# TODO: Possible to remove for REMIND coupling?
rule build_energy_totals:
    params:
        countries=config_provider("countries"),
        energy=config_provider("energy"),
    input:
        nuts3_shapes=resources("nuts3_shapes.geojson"),
        co2="data/bundle/eea/UNFCCC_v23.csv",
        swiss="data/switzerland-new_format-all_years.csv",
        swiss_transport="data/gr-e-11.03.02.01.01-cc.csv",
        idees="data/jrc-idees-2021",
        district_heat_share="data/district_heat_share.csv",
        eurostat="data/eurostat/Balances-April2023",
        eurostat_households="data/eurostat/eurostat-household_energy_balances-february_2024.csv",
    output:
        transformation_output_coke=resources("transformation_output_coke.csv"),
        energy_name=resources("energy_totals.csv"),
        co2_name=resources("co2_totals.csv"),
        transport_name=resources("transport_data.csv"),
        district_heat_share=resources("district_heat_share.csv"),
        heating_efficiencies=resources("heating_efficiencies.csv"),
    threads: 16
    resources:
        mem_mb=10000,
    log:
        logs("build_energy_totals.log"),
    benchmark:
        benchmarks("build_energy_totals")
    group:
        "iy"
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/build_energy_totals.py"

# Paths have been modified, but script is unchanged
rule build_population_weighted_energy_totals:
    params:
        snapshots=config_provider("snapshots"),
    input:
        energy_totals=resources("{kind}_totals.csv"),
        clustered_pop_layout=SCENARIO_RESOURCES
        + "i{iteration}/y{year}/pop_layout_elec_s{simpl}_{clusters}.csv",
    output:
        SCENARIO_RESOURCES
        + "i{iteration}/y{year}/pop_weighted_{kind}_totals_s{simpl}_{clusters}.csv",
    threads: 1
    resources:
        mem_mb=2000,
    log:
        SCENARIO_LOGS
        + "i{iteration}/y{year}/build_population_weighted_{kind}_totals_s{simpl}_{clusters}.log",
    group:
        "iy"
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/build_population_weighted_energy_totals.py"

# Unchanged
def heat_demand_cutout(wildcards):
    c = config_provider("sector", "heat_demand_cutout")(wildcards)
    if c == "default":
        return (
            "cutouts/"
            + CDIR
            + config_provider("atlite", "default_cutout")(wildcards)
            + ".nc"
        )
    else:
        return "cutouts/" + CDIR + c + ".nc"

# Paths have been modified, but script is unchanged
rule build_temperature_profiles:
    params:
        snapshots=config_provider("snapshots"),
        drop_leap_day=config_provider("enable", "drop_leap_day"),
    input:
        pop_layout=resources("pop_layout_total.nc"),
        regions_onshore=SCENARIO_RESOURCES
        + "i{iteration}/y{year}/regions_onshore_elec_s{simpl}_{clusters}.geojson",
        cutout=heat_demand_cutout,
    output:
        temp_soil=SCENARIO_RESOURCES
        + "i{iteration}/y{year}/temp_soil_total_elec_s{simpl}_{clusters}.nc",
        temp_air=SCENARIO_RESOURCES
        + "i{iteration}/y{year}/temp_air_total_elec_s{simpl}_{clusters}.nc",
    resources:
        mem_mb=20000,
    threads: 8
    log:
        SCENARIO_LOGS
        + "i{iteration}/y{year}/build_temperature_profiles_{simpl}_{clusters}.log",
    benchmark:
        SCENARIO_BENCHMARKS
        + "i{iteration}/y{year}/build_temperature_profiles/s{simpl}_{clusters}"
    group:
        "iy"
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/build_temperature_profiles.py"

# Paths have been modified, but script is unchanged
rule build_transport_demand:
    params:
        snapshots=config_provider("snapshots"),
        drop_leap_day=config_provider("enable", "drop_leap_day"),
        sector=config_provider("sector"),
        energy_totals_year=config_provider("energy", "energy_totals_year"),
    input:
        clustered_pop_layout=SCENARIO_RESOURCES
        + "i{iteration}/y{year}/pop_layout_elec_s{simpl}_{clusters}.csv",
        pop_weighted_energy_totals=SCENARIO_RESOURCES
        + "i{iteration}/y{year}/pop_weighted_energy_totals_s{simpl}_{clusters}.csv",
        transport_data=resources("transport_data.csv"),
        traffic_data_KFZ="data/bundle/emobility/KFZ__count",
        traffic_data_Pkw="data/bundle/emobility/Pkw__count",
        temp_air_total=SCENARIO_RESOURCES
        + "i{iteration}/y{year}/temp_air_total_elec_s{simpl}_{clusters}.nc",
    output:
        transport_demand=SCENARIO_RESOURCES
        + "i{iteration}/y{year}/transport_demand_s{simpl}_{clusters}.csv",
        transport_data=SCENARIO_RESOURCES
        + "i{iteration}/y{year}/transport_data_s{simpl}_{clusters}.csv",
        avail_profile=SCENARIO_RESOURCES
        + "i{iteration}/y{year}/avail_profile_s{simpl}_{clusters}.csv",
        dsm_profile=SCENARIO_RESOURCES
        + "i{iteration}/y{year}/dsm_profile_s{simpl}_{clusters}.csv", 
    threads: 1
    resources:
        mem_mb=2000,
    log:
        SCENARIO_LOGS
        + "i{iteration}/y{year}/build_transport_demand_{simpl}_{clusters}.log",
    group:
        "iy"
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/build_transport_demand.py"

# New rule for REMIND coupling
# Currently only adds EVs to the network
rule prepare_sector_network_REMIND:
    params:
        remind_settings=config_provider("remind_coupling", "sector_coupling"),
    input:
        # Network output from rule prepare_network
        network=SCENARIO_RESOURCES
        + "i{iteration}/y{year}/networks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}.nc",
        # REMIND gdx with EV load
        remind_data=SCENARIO_RESOURCES + "i{iteration}/REMIND2PyPSAEUR.gdx",
        # Transport-related input files
        transport_demand=SCENARIO_RESOURCES
        + "i{iteration}/y{year}/transport_demand_s{simpl}_{clusters}.csv",
        transport_data=SCENARIO_RESOURCES
        + "i{iteration}/y{year}/transport_data_s{simpl}_{clusters}.csv",
        avail_profile=SCENARIO_RESOURCES
        + "i{iteration}/y{year}/avail_profile_s{simpl}_{clusters}.csv",
        dsm_profile=SCENARIO_RESOURCES
        + "i{iteration}/y{year}/dsm_profile_s{simpl}_{clusters}.csv",
        clustered_pop_layout=SCENARIO_RESOURCES
        + "i{iteration}/y{year}/pop_layout_elec_s{simpl}_{clusters}.csv",
    output:
        # Network with sector coupling (suffix sc)
        network=SCENARIO_RESOURCES
        + "i{iteration}/y{year}/networks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_sc.nc",
    threads: 1
    resources:
        mem_mb=2000,
    log:
        SCENARIO_LOGS
        + "i{iteration}/y{year}/prepare_sector_network_REMIND_{simpl}_{clusters}_ec_l{ll}_{opts}_sc.log",
    benchmark:
        SCENARIO_BENCHMARKS
        + "i{iteration}/y{year}/prepare_sector_network_REMIND/s{simpl}_{clusters}_ec_l{ll}_{opts}_sc"
    group:
        "iy"
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/prepare_sector_network_REMIND.py"