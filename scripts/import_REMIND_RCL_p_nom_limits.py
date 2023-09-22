# -*- coding: utf-8 -*-
# %%
import logging

import numpy as np
import pandas as pd
from _helpers import configure_logging, get_technology_mapping, read_remind_data
from gams import transfer as gt

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "import_REMIND_RCL_p_nom_limits",
            configfiles="config.remind.yaml",
            scenario="no_scenario",
            iteration="1",
            simpl="",
            opts="1H-RCL-Ep205",
            clusters="4",
            ll="copt",
            year="2070",
        )
    configure_logging(snakemake)

    logger.info("Loading REMIND data ...")
    min_capacities = read_remind_data(
        snakemake.input["remind_data"],
        "p32_preInvCapAvg",
        rename_columns={
            "ttot": "year",
            "all_regi": "region_REMIND",
            "all_te": "remind_technology",
        },
    )
    min_capacities["value"] *= 1e6  # unit conversion: TW to MW
    min_capacities = min_capacities.loc[
        min_capacities["year"] == snakemake.wildcards["year"]
    ]

    # Capacity limits apply to general technology groups, not REMIND technologies
    logger.info(
        "Aggregating min capacities (p_nom_min) by general technology groups and REMIND regions ..."
    )
    min_capacities["general_technology"] = min_capacities["remind_technology"].map(
        {
            k: v[0]
            for k, v in get_technology_mapping(
                snakemake.input["technology_mapping"],
                source="REMIND-EU",
                target="general",
            ).items()
        }
    )
    min_capacities = (
        min_capacities.groupby(["region_REMIND", "general_technology"])["value"]
        .sum()
        .to_frame("p_nom_min")
        .round(2)
        .where(lambda x: x > 0)
        .dropna()
        .reset_index()
    )

    # Export
    logger.info(f"Exporting data to {snakemake.output['RCL_p_nom_limits']}")
    min_capacities.to_csv(snakemake.output["RCL_p_nom_limits"], index=False)
