# -*- coding: utf-8 -*-
# %%
import logging

import country_converter as coco
import numpy as np
import pandas as pd
from _helpers import configure_logging
from gams import transfer as gt

logger = logging.getLogger(__name__)

# TODO mapping hard-coded for now and redundant with extract_coupling_parameters.py
# TODO remove redundancy and outsource into external file
# Only Generation technologies (PyPSA "generator" "carriers")
# Use a two step mapping approach between PyPSA-EUR and REMIND:
# First mapping is aggregating PyPSA-EUR technologies to general technologies
# Second mapping is disaggregating general technologies to REMIND technologies
map_pypsaeur_to_general = {
    "CCGT": "CCGT",
    "OCGT": "OCGT",
    "biomass": "biomass",
    "coal": "all_coal",
    "offwind": "wind_offshore",
    "oil": "oil",
    "onwind": "wind_onshore",
    "solar": "solar_pv",
    "nuclear": "nuclear",
    "hydro": "hydro",
    "lignite": "all_coal",
    "ror": "hydro",
}

map_general_to_remind = {
    "CCGT": ["ngcc", "ngccc", "gaschp"],
    "OCGT": ["ngt"],
    "biomass": ["biochp", "bioigcc", "bioigccc"],
    "all_coal": ["igcc", "igccc", "pc", "coalchp"],
    "nuclear": ["tnrs", "fnrs"],
    "oil": ["dot"],
    "solar_pv": ["spv"],
    "wind_offshore": ["windoff"],
    "wind_onshore": ["wind"],
    "hydro": ["hydro"],
}

# Inverted maps required here
map_remind_to_general = {lv: k for k, v in map_general_to_remind.items() for lv in v}
map_general_to_pypsaeur = (
    pd.DataFrame(
        {
            "PyPSA-EUR": map_pypsaeur_to_general.keys(),
            "technology_group": map_pypsaeur_to_general.values(),
        }
    )
    .groupby("technology_group")["PyPSA-EUR"]
    .apply(list)
    .to_dict()
)


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
            year="2050",
        )
    configure_logging(snakemake)

    # Create region mapping by loading the original mapping from REMIND-EU from file
    # and then mapping ISO 3166-1 alpha-3 country codes to PyPSA-EUR ISO 3166-1 alpha-2 country codes
    logger.info("Loading region mapping and adding Kosovo (KV) manually.")
    region_mapping = pd.read_csv(snakemake.input["region_mapping"], sep=";").rename(
        columns={"RegionCode": "REMIND-EU"}
    )

    region_mapping["PyPSA-EUR"] = coco.convert(
        names=region_mapping["CountryCode"], to="ISO2"
    )
    region_mapping = region_mapping[["PyPSA-EUR", "REMIND-EU"]]

    # Append Kosovo to region mapping, not present in standard mapping and uses non-standard "KV" in PyPSA-EUR
    region_mapping = pd.concat(
        [
            region_mapping,
            pd.DataFrame(
                {
                    "REMIND-EU": "NES",
                    "PyPSA-EUR": "KV",
                },
                columns=["PyPSA-EUR", "REMIND-EU"],
                index=[0],
            ),
        ]
    ).reset_index(drop=True)

    # Limit mapping to countries modelled with PyPSA-EUR
    region_mapping = region_mapping.loc[
        region_mapping["PyPSA-EUR"].isin(snakemake.config["countries"])
    ]

    region_mapping = (
        region_mapping.groupby("REMIND-EU")["PyPSA-EUR"].apply(list).to_dict()
    )

    logger.info("Loading REMIND data ...")
    remind_data = gt.Container(snakemake.input["remind_data"])
    min_capacities = remind_data["p32_preInvCap"].records.rename(
        columns={
            "tPy32": "year",
            "regPy32": "region_REMIND",
            "tePy32": "remind_technology",
        }
    )
    min_capacities["value"] *= 1e6  # unit conversion: TW to MW
    min_capacities = min_capacities.loc[
        min_capacities["year"] == snakemake.wildcards["year"]
    ]

    # Capacity limits apply to general technology groups, not REMIND technologies
    logger.info(
        "Aggregating min capacities (p_nom_min) by general technology groups and REMIND regions ..."
    )
    min_capacities["technology_group"] = min_capacities["remind_technology"].map(
        map_remind_to_general
    )
    min_capacities = (
        min_capacities.groupby(["region_REMIND", "technology_group"])["value"]
        .sum()
        .to_frame("p_nom_min")
        .reset_index()
    )

    # Add groups of (PyPSA-EUR) countries + carriers to which the p_nom limits should apply
    min_capacities["carriers_PyPSA-EUR"] = min_capacities["technology_group"].map(
        map_general_to_pypsaeur
    )
    min_capacities["country_PyPSA-EUR"] = (
        min_capacities["region_REMIND"].astype(str).map(region_mapping)
    )

    # Export
    logger.info(f"Exporting data to {snakemake.output['RCL_p_nom_limits']}")
    min_capacities.to_csv(snakemake.output["RCL_p_nom_limits"], index=False)
