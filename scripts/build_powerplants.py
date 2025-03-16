# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2017-2024 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT

# coding: utf-8
# %%
"""
Retrieves conventional powerplant capacities and locations from
`powerplantmatching <https://github.com/PyPSA/powerplantmatching>`_, assigns
these to buses and creates a ``.csv`` file. It is possible to amend the
powerplant database with custom entries provided in
``data/custom_powerplants.csv``.
Lastly, for every substation, powerplants with zero-initial capacity can be added for certain fuel types automatically.

Relevant Settings
-----------------

.. code:: yaml

    electricity:
      powerplants_filter:
      custom_powerplants:
      everywhere_powerplants:

.. seealso::
    Documentation of the configuration file ``config/config.yaml`` at
    :ref:`electricity`

Inputs
------

- ``networks/base.nc``: confer :ref:`base`.
- ``data/custom_powerplants.csv``: custom powerplants in the same format as `powerplantmatching <https://github.com/PyPSA/powerplantmatching>`_ provides

Outputs
-------

- ``resource/powerplants.csv``: A list of conventional power plants (i.e. neither wind nor solar) with fields for name, fuel type, technology, country, capacity in MW, duration, commissioning year, retrofit year, latitude, longitude, and dam information as documented in the `powerplantmatching README <https://github.com/PyPSA/powerplantmatching/blob/master/README.md>`_; additionally it includes information on the closest substation/bus in ``networks/base.nc``.

    .. image:: img/powerplantmatching.png
        :scale: 30 %

    **Source:** `powerplantmatching on GitHub <https://github.com/PyPSA/powerplantmatching>`_

Description
-----------

The configuration options ``electricity: powerplants_filter`` and ``electricity: custom_powerplants`` can be used to control whether data should be retrieved from the original powerplants database or from custom amendmends. These specify `pandas.query <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.query.html>`_ commands.
In addition the configuration option ``electricity: everywhere_powerplants`` can be used to place powerplants with zero-initial capacity of certain fuel types at all substations.

1. Adding all powerplants from custom:

    .. code:: yaml

        powerplants_filter: false
        custom_powerplants: true

2. Replacing powerplants in e.g. Germany by custom data:

    .. code:: yaml

        powerplants_filter: Country not in ['Germany']
        custom_powerplants: true

    or

    .. code:: yaml

        powerplants_filter: Country not in ['Germany']
        custom_powerplants: Country in ['Germany']


3. Adding additional built year constraints:

    .. code:: yaml

        powerplants_filter: Country not in ['Germany'] and YearCommissioned <= 2015
        custom_powerplants: YearCommissioned <= 2015

4. Adding powerplants at all substations for 4 conventional carrier types:

    .. code:: yaml

        everywhere_powerplants: ['Natural Gas', 'Coal', 'nuclear', 'OCGT']
"""

import itertools
import logging

import numpy as np
import pandas as pd
import powerplantmatching as pm
import pypsa
from _helpers import configure_logging, set_scenario_config, get_region_mapping
from powerplantmatching.export import map_country_bus

logger = logging.getLogger(__name__)


def add_custom_powerplants(ppl, custom_powerplants, custom_ppl_query=False):
    if not custom_ppl_query:
        return ppl
    add_ppls = pd.read_csv(custom_powerplants, dtype={"bus": "str"})
    if isinstance(custom_ppl_query, str):
        add_ppls.query(custom_ppl_query, inplace=True)
    return pd.concat(
        [ppl, add_ppls], sort=False, ignore_index=True, verify_integrity=True
    )


def add_everywhere_powerplants(ppl, substations, everywhere_powerplants):
    # Create a dataframe with "everywhere_powerplants" of stated carriers at the location of all substations
    everywhere_ppl = (
        pd.DataFrame(
            itertools.product(substations.index.values, everywhere_powerplants),
            columns=["substation_index", "Fueltype"],
        ).merge(
            substations[["x", "y", "country"]],
            left_on="substation_index",
            right_index=True,
        )
    ).drop(columns="substation_index")

    # PPL uses different columns names compared to substations dataframe -> rename
    everywhere_ppl = everywhere_ppl.rename(
        columns={"x": "lon", "y": "lat", "country": "Country"}
    )

    # Add default values for the powerplants
    everywhere_ppl["Name"] = (
        "Automatically added everywhere-powerplant " + everywhere_ppl.Fueltype
    )
    everywhere_ppl["Set"] = "PP"
    everywhere_ppl["Technology"] = everywhere_ppl["Fueltype"]
    everywhere_ppl["Capacity"] = 0.0

    # Assign plausible values for the commissioning and decommissioning years
    # required for multi-year models
    everywhere_ppl["DateIn"] = ppl["DateIn"].min()
    everywhere_ppl["DateOut"] = ppl["DateOut"].max()

    # NaN values for efficiency will be replaced by the generic efficiency by attach_conventional_generators(...) in add_electricity.py later
    everywhere_ppl["Efficiency"] = np.nan

    return pd.concat(
        [ppl, everywhere_ppl], sort=False, ignore_index=True, verify_integrity=True
    )


def replace_natural_gas_technology(df):
    mapping = {
        "Steam Turbine": "CCGT",
        "Combustion Engine": "OCGT",
        "Not Found": "CCGT",
    }
    tech = df.Technology.replace(mapping).fillna("CCGT")
    return df.Technology.mask(df.Fueltype == "Natural Gas", tech)


def replace_natural_gas_fueltype(df):
    return df.Fueltype.mask(
        (df.Technology == "OCGT") | (df.Technology == "CCGT"), "Natural Gas"
    )


# %%
if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("build_powerplants",
                                   scenario = "PyPSA_PkBudg1000_DEU_oneNode_RCLgenOnly_noAdjCost_2025-03-13_16.21.26",
                                   iteration = 3,
                                   year = 2150)
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    n = pypsa.Network(snakemake.input.base_network)
    countries = snakemake.params.countries

    # Read powerplants from snakemake.input["powerplants"]
    ppl = pd.read_csv(snakemake.input.powerplants, index_col=0)
    
    ppl = (
        ppl
        .powerplant.fill_missing_decommissioning_years()
        .powerplant.convert_country_to_alpha2()
        .query('Fueltype not in ["Solar", "Wind"] and Country in @countries')
        .assign(Technology=replace_natural_gas_technology)
        .assign(Fueltype=replace_natural_gas_fueltype)
    )

    # Correct bioenergy for countries where possible
    opsd = pm.data.OPSD_VRE().powerplant.convert_country_to_alpha2()
    opsd = opsd.query('Country in @countries and Fueltype == "Bioenergy"')
    opsd["Name"] = "Biomass"
    available_countries = opsd.Country.unique()
    ppl = ppl.query('not (Country in @available_countries and Fueltype == "Bioenergy")')
    ppl = pd.concat([ppl, opsd])

    ppl_query = snakemake.params.powerplants_filter
    if isinstance(ppl_query, str):
        ppl.query(ppl_query, inplace=True)

    # add carriers from own powerplant files:
    custom_ppl_query = snakemake.params.custom_powerplants
    ppl = add_custom_powerplants(
        ppl, snakemake.input.custom_powerplants, custom_ppl_query
    )

    if countries_wo_ppl := set(countries) - set(ppl.Country.unique()):
        logging.warning(f"No powerplants known in: {', '.join(countries_wo_ppl)}")

    # REMIND coupling specific: Filter powerplants for the year, including those with no DateOut date
    year = int(snakemake.wildcards.year)
    # Don't filter hydro (dealt with separately in attach_hydro)
    ppl = ppl.query(
        "(Fueltype == 'Hydro') or "
        "(DateIn <= @year and "
        "(DateOut >= @year or DateOut.isna()))"
    )

    # Read p_nom_limits 
    p_nom_limits = pd.read_csv(snakemake.input.RCL_p_nom_limits, index_col=0).reset_index()
    
    # Map fuel type and technology to p_nom_limits
    map_fueltype_p_nom = {
        "Lignite": "coal & lignite",
        "Hard Coal": "coal & lignite",
        "Bioenergy": "biomass",
        "Nuclear": "nuclear",
        "Oil": "oil",
    }
    
    # Map technology to p_nom_limits
    map_tech_p_nom = {
        "CCGT": "CCGT",
        "OCGT": "OCGT"
    }
    
    # Apply mappings to new column
    ppl["technology_group"] = ppl.Fueltype.map(map_fueltype_p_nom)
    # If type is NaN, map technology and fill in the missing values
    ppl["technology_group"] = ppl["technology_group"].fillna(ppl.Technology.map(map_tech_p_nom))
    
    # Map Country to REMIND region
    region_mapping = get_region_mapping(
        snakemake.input["region_mapping"], source="PyPSA-EUR", target="REMIND-EU", flatten = True
    )
    ppl["region_REMIND"] = ppl["Country"].map(region_mapping)
    
    diff = (ppl[["region_REMIND", "technology_group", "Capacity"]]
            .groupby(["region_REMIND", "technology_group"])
            .sum()
            .reset_index())
    
    # Merge with p_nom_limits and determine difference
    diff = diff.merge(p_nom_limits)
    diff["diff_abs"] = diff["Capacity"] - diff["p_nom_min"]
    diff["diff_factor"] = diff["Capacity"] / diff["p_nom_min"]
        
    # Adjust powerplant database
    ppl = ppl.merge(
        diff[["technology_group", "diff_factor"]],
        left_on="technology_group",
        right_on="technology_group",
        how="left")
    
    # If diff_factor is less than one, no need to change anything
    # If diff_factor is greater than one, adjust capacity of each powerplant
    ppl.loc[ppl["diff_factor"] > 1, "Capacity"] = ppl["Capacity"] / ppl["diff_factor"]
    
    # Remove columns
    ppl = ppl.drop(columns = ["technology_group", "region_REMIND", "diff_factor"])
    
    # Logging
    logging.info(f"Capacity of powerplants adjusted to RCL constraints:")
    # Show log for each technology_group, posting the absolute difference
    for tech_group in diff["technology_group"].unique():
        diff_tech = diff.loc[diff["technology_group"] == tech_group]
        before = round(diff_tech["Capacity"].values[0]/1E3, 2)
        after = round(diff_tech["p_nom_min"].values[0]/1E3, 2)
        if (after < before):
            logging.info(f"Adjusting powerplants database for {tech_group} from {before} GW to {after} GW")
        else:
            logging.info(f"No adjustment to powerplants database for {tech_group} with {before} GW")
    
    # Update p_nom_limits
    p_nom_limits_updated = p_nom_limits.merge(diff, how = "left")
    # If diff_factor is > 1, all capacities are set through existing powerplants
    p_nom_limits_updated.loc[
        p_nom_limits_updated["diff_factor"] > 1, "p_nom_min"
        ] = 0
    # If diff_factor is < 1, capacities are set through updated RCL constraint
    p_nom_limits_updated.loc[
        p_nom_limits_updated["diff_factor"] < 1, "p_nom_min"
        ] = p_nom_limits_updated["p_nom_min"] - p_nom_limits_updated["Capacity"]
    p_nom_limits_updated = p_nom_limits_updated[["region_REMIND", "technology_group", "p_nom_min"]]
    
    p_nom_limits_updated.to_csv(snakemake.output.RCL_p_nom_limits_updated, index = False)
    
    # Add "everywhere powerplants" to all bus locations
    ppl = add_everywhere_powerplants(
        ppl, n.buses, snakemake.params.everywhere_powerplants
    )

    ppl = ppl.dropna(subset=["lat", "lon"])
    ppl = map_country_bus(ppl, n.buses)

    bus_null_b = ppl["bus"].isnull()
    if bus_null_b.any():
        logging.warning(
            f"Couldn't find close bus for {bus_null_b.sum()} powerplants. "
            "Removing them from the powerplants list."
        )
        ppl = ppl[~bus_null_b]

    # TODO: This has to fixed in PPM, some powerplants are still duplicated
    cumcount = ppl.groupby(["bus", "Fueltype"]).cumcount() + 1
    ppl.Name = ppl.Name.where(cumcount == 1, ppl.Name + " " + cumcount.astype(str))

    ppl.reset_index(drop=True).to_csv(snakemake.output[0])

# %%
