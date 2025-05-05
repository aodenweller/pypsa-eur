# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
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
        cutout=lambda w: input_cutout(w),
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
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/build_population_layouts.py"


# Unchanged
rule build_clustered_population_layouts:
    input:
        pop_layout_total=resources("pop_layout_total.nc"),
        pop_layout_urban=resources("pop_layout_urban.nc"),
        pop_layout_rural=resources("pop_layout_rural.nc"),
        regions_onshore=resources("regions_onshore_base_s_{clusters}.geojson"),
        cutout=lambda w: input_cutout(w),
    output:
        clustered_pop_layout=resources("pop_layout_base_s_{clusters}.csv"),
    log:
        logs("build_clustered_population_layouts_s_{clusters}.log"),
    resources:
        mem_mb=10000,
    benchmark:
        benchmarks("build_clustered_population_layouts/s_{clusters}")
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/build_clustered_population_layouts.py"


# Unchanged
rule build_simplified_population_layouts:
    input:
        pop_layout_total=resources("pop_layout_total.nc"),
        pop_layout_urban=resources("pop_layout_urban.nc"),
        pop_layout_rural=resources("pop_layout_rural.nc"),
        regions_onshore=resources("regions_onshore_base_s.geojson"),
        cutout=lambda w: input_cutout(w),
    output:
        clustered_pop_layout=resources("pop_layout_base_s.csv"),
    resources:
        mem_mb=10000,
    log:
        logs("build_simplified_population_layouts_s"),
    benchmark:
        benchmarks("build_simplified_population_layouts/s")
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/build_clustered_population_layouts.py"


# Unchanged
rule build_temperature_profiles:
    params:
        snapshots=config_provider("snapshots"),
        drop_leap_day=config_provider("enable", "drop_leap_day"),
    input:
        pop_layout=resources("pop_layout_total.nc"),
        regions_onshore=resources("regions_onshore_base_s_{clusters}.geojson"),
        cutout=lambda w: input_cutout(
            w, config_provider("sector", "heat_demand_cutout")(w)
        ),
    output:
        temp_soil=resources("temp_soil_total_base_s_{clusters}.nc"),
        temp_air=resources("temp_air_total_base_s_{clusters}.nc"),
    resources:
        mem_mb=20000,
    threads: 8
    log:
        logs("build_temperature_profiles_total_s_{clusters}.log"),
    benchmark:
        benchmarks("build_temperature_profiles/total_{clusters}")
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/build_temperature_profiles.py"


# Unchanged
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
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/build_energy_totals.py"


# TODO: Incorporate later
rule build_salt_cavern_potentials:
    input:
        salt_caverns="data/bundle/h2_salt_caverns_GWh_per_sqkm.geojson",
        regions_onshore=resources("regions_onshore_base_s_{clusters}.geojson"),
        regions_offshore=resources("regions_offshore_base_s_{clusters}.geojson"),
    output:
        h2_cavern_potential=resources("salt_cavern_potentials_s_{clusters}.csv"),
    threads: 1
    resources:
        mem_mb=2000,
    log:
        logs("build_salt_cavern_potentials_s_{clusters}.log"),
    benchmark:
        benchmarks("build_salt_cavern_potentials_s_{clusters}")
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/build_salt_cavern_potentials.py"

# Unchanged
rule build_population_weighted_energy_totals:
    params:
        snapshots=config_provider("snapshots"),
        drop_leap_day=config_provider("enable", "drop_leap_day"),
    input:
        energy_totals=resources("{kind}_totals.csv"),
        clustered_pop_layout=resources("pop_layout_base_s_{clusters}.csv"),
    output:
        resources("pop_weighted_{kind}_totals_s_{clusters}.csv"),
    threads: 1
    resources:
        mem_mb=2000,
    log:
        logs("build_population_weighted_{kind}_totals_{clusters}.log"),
    benchmark:
        benchmarks("build_population_weighted_{kind}_totals_{clusters}")
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/build_population_weighted_energy_totals.py"


rule build_transport_demand:
    params:
        snapshots=config_provider("snapshots"),
        drop_leap_day=config_provider("enable", "drop_leap_day"),
        sector=config_provider("sector"),
        energy_totals_year=config_provider("energy", "energy_totals_year"),
    input:
        clustered_pop_layout=resources("pop_layout_base_s_{clusters}.csv"),
        pop_weighted_energy_totals=resources(
            "pop_weighted_energy_totals_s_{clusters}.csv"
        ),
        transport_data=resources("transport_data.csv"),
        traffic_data_KFZ="data/bundle/emobility/KFZ__count",
        traffic_data_Pkw="data/bundle/emobility/Pkw__count",
        temp_air_total=resources("temp_air_total_base_s_{clusters}.nc"),
    output:
        transport_demand=resources("transport_demand_s_{clusters}.csv"),
        transport_data=resources("transport_data_s_{clusters}.csv"),
        avail_profile=resources("avail_profile_s_{clusters}.csv"),
        dsm_profile=resources("dsm_profile_s_{clusters}.csv"),
    threads: 1
    resources:
        mem_mb=2000,
    log:
        logs("build_transport_demand_s_{clusters}.log"),
    benchmark:
        benchmarks("build_transport_demand/s_{clusters}")
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/build_transport_demand.py"


# New rule for REMIND coupling
# Currently only adds EVs to the network
# rule prepare_sector_network_REMIND:
#     params:
#         remind_settings=config_provider("remind_coupling", "sector_coupling"),
#     input:
#         # Network output from rule prepare_network
#         network=SCENARIO_RESOURCES
#         + "i{iteration}/y{year}/networks/base_s_{clusters}_elec_{opts}.nc",
#         # REMIND gdx with EV load
#         remind_data=SCENARIO_RESOURCES + "i{iteration}/REMIND2PyPSAEUR.gdx",
#         # Transport-related input files
#         transport_demand=resources("transport_demand_s_{clusters}.csv"),
#         transport_data=resources("transport_data_s_{clusters}.csv"),
#         avail_profile=resources("avail_profile_s_{clusters}.csv"),
#         dsm_profile=resources("dsm_profile_s_{clusters}.csv"),
#     output:
#         # Network with sector coupling (suffix sc)
#         network=SCENARIO_RESOURCES
#         + "i{iteration}/y{year}/networks/base_s_{clusters}_elec_{opts}_sc.nc",
#     threads: 1
#     resources:
#         mem_mb=2000,
#     log:
#         SCENARIO_LOGS
#         + "i{iteration}/y{year}/prepare_sector_network_REMIND/s_{clusters}_elec_{opts}_sc.log",
#     benchmark:
#         SCENARIO_BENCHMARKS
#         + "i{iteration}/y{year}/prepare_sector_network_REMIND/s_{clusters}_elec_{opts}_sc"
#     group:
#         "iy"
#     conda:
#         "../envs/environment.yaml"
#     script:
#         "../scripts/prepare_sector_network_REMIND.py"