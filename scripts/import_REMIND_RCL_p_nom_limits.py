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
            configfiles="config/config.remind.yaml",
            scenario="PyPSA_PkBudg1000_DEU_newLoad_h2stor_2025-02-17_20.41.40",
            iteration="8",
            simpl="",
            opts="3H-RCL-Ep174.2",
            clusters="4",
            ll="copt",
            year="2050",
        )
    configure_logging(snakemake)
    
#%%
    logger.info("Loading REMIND pre-investment capacities...")
    min_capacities = read_remind_data(
        snakemake.input["remind_data"],
        "p32_preInvCapAvg",
        rename_columns={
            "ttot": "year",
            "all_regi": "region_REMIND",
            "all_te": "remind_technology",
        },
    )
    # Unit conversion: TW to MW (for generators and links), TWh to MWh (for stores)
    min_capacities["value"] *= 1e6
    min_capacities = min_capacities.loc[
        min_capacities["year"] == snakemake.wildcards["year"]
    ]
    # Special treatment for technologies that are links in PyPSA-Eur
    # In REMIND capacities are w.r.t output capacity, in PyPSA-Eur to input capacity
    # Need to adjust capacities by the efficiency for these technologies
    logger.info(
        "Adjusting REMIND pre-investment capacities to input capacity for link technologies ..."
    )
    efficiencies = read_remind_data(
        snakemake.input["remind_data"],
        "pm_eta_conv",
        rename_columns={
            "tall": "year",
            "all_regi": "region_REMIND",
            "all_te": "remind_technology",
        },
    )
    efficiencies = efficiencies.loc[
        efficiencies["year"] == snakemake.wildcards["year"],
    ]
    # Only include regions that are in the min_capacities data
    efficiencies = efficiencies.loc[
        efficiencies["region_REMIND"].isin(min_capacities["region_REMIND"].unique())
    ]
    # For remind_technology elh2 and h2turb in min_capacities, divide by efficiency
    # to get input capacities
    remind_technologies = ["elh2", "h2turb"]
    mask = min_capacities["remind_technology"].isin(remind_technologies)
    min_capacities.loc[mask, "value"] /= efficiencies.loc[
        efficiencies["remind_technology"].isin(remind_technologies), "value"
    ].values
    
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

# %%
