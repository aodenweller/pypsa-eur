# -*- coding: utf-8 -*-
import logging

import pandas as pd
from _helpers import (
    configure_logging,
    get_region_mapping,
    mock_snakemake,
    read_remind_data,
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "import_REMIND_load",
            configfiles="config.remind.yaml",
            simpl="",
            clusters="4",
            ll="copt",
            opts="1H-RCL-Ep7.0",
            year="2070",
            iteration="1",
        )

    configure_logging(snakemake)

    # Load original load timeseries from PyPSA-EUR
    load = pd.read_csv(snakemake.input["load_timeseries"], index_col=0)

    # Load REMIND-EU demand data
    demand = read_remind_data(
        snakemake.input["remind_data"],
        "v32_usableSeDisp",
        rename_columns={
            "all_enty": "sector",
            "ttot": "year",
            "all_regi": "region",
            "level": "value",
        },
    )
    demand = demand.loc[
        (demand["sector"] == "seel") & (demand["year"] == snakemake.wildcards["year"])
    ]
    demand["value"] *= 1e6 * 8760  # Convert from TWa to MWh
    demand = demand.set_index(["region"])["value"]

    region_mapping = (
        pd.DataFrame.from_dict(
            get_region_mapping(
                snakemake.input["region_mapping"],
                source="PyPSA-EUR",
                target="REMIND-EU",
            ),
            "index",
        )
        .reset_index()
        .rename(columns={"index": "PyPSA-EUR", 0: "REMIND-EU"})
    )

    # Calculate load from PyPSA-EUR load-timeseries for REMIND-EU regions
    regional_load = (
        region_mapping.join(load.sum(axis="rows").rename("annual_load"), on="PyPSA-EUR")
        .groupby("REMIND-EU")["annual_load"]
        .sum()
    )

    # Factor by which PyPSA-EUR loads have to be scaled to match REMIND-EU demand
    load_scaling_factor = (
        (demand / regional_load).dropna().rename("load_scaling_factor")
    )
    load_scaling_factor = region_mapping.join(
        load_scaling_factor,
        on="REMIND-EU",
        how="inner",
    ).set_index("PyPSA-EUR")["load_scaling_factor"]

    # Scale load timeseries
    scaled_load = load * load_scaling_factor

    # Check whether differences after scaling is negligible
    if not all(
        (
            demand
            - region_mapping.join(
                scaled_load.sum(axis="rows").rename("annual_load"), on="PyPSA-EUR"
            )
            .groupby("REMIND-EU")["annual_load"]
            .sum()
        ).dropna()
        < 1e-5
    ):
        logger.warning(
            "Scaled load values do not seem to match REMIND-EU demand values"
        )

    scaled_load.to_csv(snakemake.output["load_timeseries"])
