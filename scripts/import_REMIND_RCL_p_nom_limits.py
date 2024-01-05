# -*- coding: utf-8 -*-
# %%
import logging

import numpy as np
import pandas as pd
from _helpers import configure_logging, get_technology_mapping, read_remind_data

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

    # Map REMIND technologies to PyPSA-Eur technologies and PyPSA-Eur technology groups
    technology_mapping = get_technology_mapping(
        snakemake.input["technology_cost_mapping"], group_technologies=True
    )

    # drop duplicate rows with REMIND-EU technology, these can in principle happen if a REMIND technology is assigned multiple PyPSA-Eur technologies
    # Since the PyPSA-Eur technologies should be part of the same technology_group, it is safe to drop the duplicates
    # e.g. "spv" (REMIND) -> ["solar", "solar-rooftop", "solar-utility"] (PyPSA-Eur) -> solar & "solar-rooftop & solar-utility" (technology_group)
    technology_mapping = technology_mapping.drop_duplicates(subset=["REMIND-EU"])
    min_capacities = min_capacities.merge(
        technology_mapping, left_on="remind_technology", right_on="REMIND-EU"
    )

    # Determine min_capacities by PyPSA-Eur technologies (grouped)
    # Rounding to a reasonable value and dropping all zero values
    min_capacities = (
        min_capacities.groupby(["region_REMIND", "technology_group"])["value"]
        .sum()
        .round(2)
        .where(lambda x: x > 0)
        .dropna()
        .reset_index()
    )
    min_capacities = min_capacities.rename(columns={"value": "p_nom_min"})

    # Export
    logger.info(f"Exporting data to {snakemake.output['RCL_p_nom_limits']}")
    min_capacities.to_csv(snakemake.output["RCL_p_nom_limits"], index=False)
