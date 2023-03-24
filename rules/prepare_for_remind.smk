# SPDX-FileCopyrightText: : 2023 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
#
# File for REMIND coupling by Adrian Odenweller (adrian.odenweller@pik-potsdam.de)

# This file creates all PyPSA files prior to any data input from REMIND

rule prepare_files_remind:
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
        regions=RESOURCES + "regions_onshore.geojson",
        powerplants=RESOURCES + "powerplants.csv",
        hydro_capacities=ancient("data/bundle/hydro_capacities.csv"),
        geth_hydro_capacities="data/geth2015_hydro_capacities.csv",
        load=RESOURCES + "load.csv",
        nuts3_shapes=RESOURCES + "nuts3_shapes.geojson"

# Run with: snakemake -call -s Snakefile_REMIND --config REMINDopt=prepare prepare_files_remind