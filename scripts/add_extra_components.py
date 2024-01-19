# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2017-2023 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT

# coding: utf-8
# %%
"""
Adds extra extendable components to the clustered and simplified network.

Relevant Settings
-----------------

.. code:: yaml

    costs:
        year:
        version:
        dicountrate:
        emission_prices:

    electricity:
        max_hours:
        marginal_cost:
        capital_cost:
        extendable_carriers:
            StorageUnit:
            Store:

.. seealso::
    Documentation of the configuration file ``config/config.yaml`` at :ref:`costs_cf`,
    :ref:`electricity_cf`

Inputs
------

- ``resources/costs.csv``: The database of cost assumptions for all included technologies for specific years from various sources; e.g. discount rate, lifetime, investment (CAPEX), fixed operation and maintenance (FOM), variable operation and maintenance (VOM), fuel costs, efficiency, carbon-dioxide intensity.

Outputs
-------

- ``networks/elec_s{simpl}_{clusters}_ec.nc``:


Description
-----------

The rule :mod:`add_extra_components` attaches additional extendable components to the clustered and simplified network. These can be configured in the ``config/config.yaml`` at ``electricity: extendable_carriers:``. It processes ``networks/elec_s{simpl}_{clusters}.nc`` to build ``networks/elec_s{simpl}_{clusters}_ec.nc``, which in contrast to the former (depending on the configuration) contain with **zero** initial capacity

- ``StorageUnits`` of carrier 'H2' and/or 'battery'. If this option is chosen, every bus is given an extendable ``StorageUnit`` of the corresponding carrier. The energy and power capacities are linked through a parameter that specifies the energy capacity as maximum hours at full dispatch power and is configured in ``electricity: max_hours:``. This linkage leads to one investment variable per storage unit. The default ``max_hours`` lead to long-term hydrogen and short-term battery storage units.

- ``Stores`` of carrier 'H2' and/or 'battery' in combination with ``Links``. If this option is chosen, the script adds extra buses with corresponding carrier where energy ``Stores`` are attached and which are connected to the corresponding power buses via two links, one each for charging and discharging. This leads to three investment variables for the energy capacity, charging and discharging capacity of the storage unit.
"""
import logging

import numpy as np
import pandas as pd
import pypsa
from _helpers import (
    configure_logging,
    get_region_mapping,
    get_technology_mapping,
    read_remind_data,
)
from add_electricity import load_costs, sanitize_carriers

logger = logging.getLogger(__name__)


def attach_storageunits(n, costs, extendable_carriers, max_hours):
    carriers = extendable_carriers["StorageUnit"]

    n.madd("Carrier", carriers)

    buses_i = n.buses.index

    lookup_store = {"H2": "electrolysis", "battery": "battery inverter"}
    lookup_dispatch = {"H2": "fuel cell", "battery": "battery inverter"}

    for carrier in carriers:
        roundtrip_correction = 0.5 if carrier == "battery" else 1

        n.madd(
            "StorageUnit",
            buses_i,
            " " + carrier,
            bus=buses_i,
            carrier=carrier,
            p_nom_extendable=True,
            capital_cost=costs.at[carrier, "capital_cost"],
            marginal_cost=costs.at[carrier, "marginal_cost"],
            efficiency_store=costs.at[lookup_store[carrier], "efficiency"]
            ** roundtrip_correction,
            efficiency_dispatch=costs.at[lookup_dispatch[carrier], "efficiency"]
            ** roundtrip_correction,
            max_hours=max_hours[carrier],
            cyclic_state_of_charge=True,
        )


def attach_stores(n, costs, extendable_carriers):
    carriers = extendable_carriers["Store"]

    n.madd("Carrier", carriers)

    buses_i = n.buses.index
    bus_sub_dict = {k: n.buses[k].values for k in ["x", "y", "country"]}

    if "H2" in carriers:
        h2_buses_i = n.madd("Bus", buses_i + " H2", carrier="H2", **bus_sub_dict)

        n.madd(
            "Store",
            h2_buses_i,
            bus=h2_buses_i,
            carrier="H2",
            e_nom_extendable=True,
            e_cyclic=True,
            capital_cost=costs.at["hydrogen storage underground", "capital_cost"],
        )

        n.madd(
            "Link",
            h2_buses_i + " Electrolysis",
            bus0=buses_i,
            bus1=h2_buses_i,
            carrier="H2 electrolysis",
            p_nom_extendable=True,
            efficiency=costs.at["electrolysis", "efficiency"],
            capital_cost=costs.at["electrolysis", "capital_cost"],
            marginal_cost=costs.at["electrolysis", "marginal_cost"],
        )

        n.madd(
            "Link",
            h2_buses_i + " Fuel Cell",
            bus0=h2_buses_i,
            bus1=buses_i,
            carrier="H2 fuel cell",
            p_nom_extendable=True,
            efficiency=costs.at["fuel cell", "efficiency"],
            # NB: fixed cost is per MWel
            capital_cost=costs.at["fuel cell", "capital_cost"]
            * costs.at["fuel cell", "efficiency"],
            marginal_cost=costs.at["fuel cell", "marginal_cost"],
        )

    if "battery" in carriers:
        b_buses_i = n.madd(
            "Bus", buses_i + " battery", carrier="battery", **bus_sub_dict
        )

        n.madd(
            "Store",
            b_buses_i,
            bus=b_buses_i,
            carrier="battery",
            e_cyclic=True,
            e_nom_extendable=True,
            capital_cost=costs.at["battery storage", "capital_cost"],
            marginal_cost=costs.at["battery", "marginal_cost"],
        )

        n.madd("Carrier", ["battery charger", "battery discharger"])

        n.madd(
            "Link",
            b_buses_i + " charger",
            bus0=buses_i,
            bus1=b_buses_i,
            carrier="battery charger",
            # the efficiencies are "round trip efficiencies"
            efficiency=costs.at["battery inverter", "efficiency"] ** 0.5,
            capital_cost=costs.at["battery inverter", "capital_cost"],
            p_nom_extendable=True,
            marginal_cost=costs.at["battery inverter", "marginal_cost"],
        )

        n.madd(
            "Link",
            b_buses_i + " discharger",
            bus0=b_buses_i,
            bus1=buses_i,
            carrier="battery discharger",
            efficiency=costs.at["battery inverter", "efficiency"] ** 0.5,
            p_nom_extendable=True,
            marginal_cost=costs.at["battery inverter", "marginal_cost"],
        )


def attach_hydrogen_pipelines(n, costs, extendable_carriers):
    as_stores = extendable_carriers.get("Store", [])

    if "H2 pipeline" not in extendable_carriers.get("Link", []):
        return

    assert "H2" in as_stores, (
        "Attaching hydrogen pipelines requires hydrogen "
        "storage to be modelled as Store-Link-Bus combination. See "
        "`config.yaml` at `electricity: extendable_carriers: Store:`."
    )

    # determine bus pairs
    attrs = ["bus0", "bus1", "length"]
    candidates = pd.concat(
        [n.lines[attrs], n.links.query('carrier=="DC"')[attrs]]
    ).reset_index(drop=True)

    # remove bus pair duplicates regardless of order of bus0 and bus1
    h2_links = candidates[
        ~pd.DataFrame(np.sort(candidates[["bus0", "bus1"]])).duplicated()
    ]
    h2_links.index = h2_links.apply(lambda c: f"H2 pipeline {c.bus0}-{c.bus1}", axis=1)

    # add pipelines
    n.add("Carrier", "H2 pipeline")

    n.madd(
        "Link",
        h2_links.index,
        bus0=h2_links.bus0.values + " H2",
        bus1=h2_links.bus1.values + " H2",
        p_min_pu=-1,
        p_nom_extendable=True,
        length=h2_links.length.values,
        capital_cost=costs.at["H2 pipeline", "capital_cost"] * h2_links.length,
        efficiency=costs.at["H2 pipeline", "efficiency"],
        carrier="H2 pipeline",
    )


def attach_RCL_generators(
    n,
    config,
    fp_p_nom_limits,
    fp_region_mapping,
    fp_technology_cost_mapping,
):
    """
    Add additional generators to network for the RCL constraint used in the
    REMIND-EU <-> PyPSA-EUR coupling.
    """
    p_nom_limits = pd.read_csv(fp_p_nom_limits)
    region_mapping = get_region_mapping(
        fp_region_mapping, source="REMIND-EU", target="PyPSA-Eur"
    )

    # Apply mapping from REMIND/general to PyPSA-EUR countries
    p_nom_limits["country"] = p_nom_limits["region_REMIND"].map(region_mapping)

    # Determine "carrier" which are related to the technology groups
    technology_mapping = (
        get_technology_mapping(fp_technology_cost_mapping, group_technologies=True)
        .set_index("technology_group")
        .rename(columns={"PyPSA-Eur": "carrier"})["carrier"]
        .drop_duplicates()
    )
    p_nom_limits = p_nom_limits.merge(
        technology_mapping, on="technology_group", how="left"
    )

    # Flatten country column entries such that all lists are converted into individual rows
    p_nom_limits = p_nom_limits.explode("country").explode("carrier")
    # Add country-reference to generators for mapping
    n.generators["country"] = n.generators["bus"].map(n.buses["country"])

    # Select all generators from n.generators where the combination of country and carrier can be found in p_nom_limits,
    # i.e. later a RCL constraint should be applied for
    rcl_generators = n.generators.join(
        p_nom_limits.set_index(["country", "carrier"]),
        on=["country", "carrier"],
        how="left",
        rsuffix="_rcl",
        validate="m:1",
    )
    rcl_generators = rcl_generators.dropna(
        subset="p_nom_min_rcl"
    )  # Drop all generators which are not subject to RCL constraint

    # Only consider RCL constraint for generators which are extendable
    rcl_generators = rcl_generators[rcl_generators["p_nom_extendable"] == True]

    # Modify properties of to-be-added RCL generators which differ from the original generators
    old_generators = rcl_generators.index
    rcl_generators.index = old_generators + " (RCL)"
    rcl_generators["capital_cost"] = config["capital_cost"]
    rcl_generators["p_nom_min"] = 0.0
    rcl_generators["p_nom"] = 0.0
    rcl_generators["p_nom_max"] = np.inf

    # Finally add RCL generators to network
    n.madd("Generator", rcl_generators.index, **rcl_generators)

    # Transfer time-dependent dispatch limits which are not transfered thorugh n.madd(...)
    n.pnl("Generator")["p_min_pu"] = pd.merge(
        n.pnl("Generator")["p_min_pu"],
        n.pnl("Generator")["p_min_pu"][
            old_generators.intersection(n.pnl("Generator")["p_min_pu"].columns)
        ].rename(columns=lambda x: x + " (RCL)"),
        left_index=True,
        right_index=True,
    )
    n.pnl("Generator")["p_max_pu"] = pd.merge(
        n.pnl("Generator")["p_max_pu"],
        n.pnl("Generator")["p_max_pu"][
            old_generators.intersection(n.pnl("Generator")["p_max_pu"].columns)
        ].rename(columns=lambda x: x + " (RCL)"),
        left_index=True,
        right_index=True,
    )


def attach_hydrogen_demand(
    n,
    year,
    config,
    fp_remind_data,
    fp_region_mapping,
):
    """
    Add optional H2 demand for hydrogen from electrolysis based on REMIND
    scenarios to the network.

    Each REMIND region is assigned a single shared H2 demand which is
    connected to all existing H2 buses from PyPSAEur within this region.
    The connection is made using a uni-directional link. The hydrogen
    demand is converted from annual TWa (REMIND) to MW of constant load.
    An optional H2 buffer store with configurable size can be added.
    Links and H2 buffer are added without any cost.
    """
    # map countries to REMIND regions
    # Create region mapping
    region_mapping = get_region_mapping(
        fp_region_mapping, source="PyPSA-EUR", target="REMIND-EU"
    )
    region_mapping = pd.DataFrame(region_mapping).T.reset_index()
    region_mapping.columns = ["PyPSA-EUR", "REMIND-EU"]
    region_mapping = region_mapping.set_index("PyPSA-EUR")

    # Find all H2 buses which we connect to REMIND demand bus
    original_h2_buses = n.buses[n.buses["carrier"] == "H2"][["country"]]

    # Map countries to REMIND regions
    original_h2_buses["region"] = original_h2_buses["country"].map(
        region_mapping["REMIND-EU"]
    )

    # Load H2 demand from REMIND gdx file
    h2_demand = read_remind_data(
        fp_remind_data,
        "p32_ElecH2Demand",
        rename_columns={"ttot": "year", "all_regi": "region"},
    )
    h2_demand["value"] *= 8760 * 1e6  # convert TWa to MWh
    # Restrict to relevant year and regions inside the model
    h2_demand = h2_demand.loc[
        (h2_demand["year"] == str(year))
        & (h2_demand["region"].isin(original_h2_buses["region"].unique()))
    ]

    n.buses["region"] = ""

    for idx, row in h2_demand.iterrows():
        n.add(
            "Bus",
            name=f"{row['region']} H2 demand",
            carrier="H2 demand REMIND",
        )
        # Add the region for the H2 demand bus directly to the dataframe, as they are more difficult to map later
        n.buses.loc[f"{row['region']} H2 demand", "region"] = row["region"]

        n.add(
            "Load",
            name=f"{row['region']} H2 demand REMIND",
            bus=f"{row['region']} H2 demand",
            p_set=row["value"] / 8760,
        )

        n.add(
            "Store",
            name=f"{row['region']} H2 demand buffer REMIND",
            bus=f"{row['region']} H2 demand",
            e_nom_extendable=False,
            e_cyclic=True,
            e_nom=row["value"] / 8760 * config["buffer_max_hours"],
            capital_cost=0,
            marginal_cost=0,
            carrier="H2 demand buffer REMIND",
        )

    # Connect PyPSAEur H2 buses (per node) to REMIND H2 demand buses (per region)
    for idx, row in original_h2_buses.iterrows():
        n.add(
            "Link",
            name=f"{idx} transfer to {row['region']} H2 demand REMIND",
            bus0=idx,
            bus1=f"{row['region']} H2 demand",
            p_min_pu=0,  # unidirectional, only allow flow from PyPSAEur H2 buses to REMIND demand
            p_max_pu=1,
            p_nom=h2_demand[
                "value"
            ].max(),  # no need for extendable, just allow max. throughput of max demand of any region
            p_nom_extendable=False,
            efficiency=1,
            capital_cost=0,
            marginal_cost=0,
            carrier="H2 transfer to H2 demand REMIND",
        )


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "add_extra_components",
            simpl="",
            clusters=6,
            scenario="h2demand",
            iteration=200,
            year=2050,
        )
    configure_logging(snakemake)

    n = pypsa.Network(snakemake.input.network)
    extendable_carriers = snakemake.params.extendable_carriers
    max_hours = snakemake.params.max_hours

    Nyears = n.snapshot_weightings.objective.sum() / 8760.0
    costs = load_costs(
        snakemake.input.tech_costs, snakemake.params.costs, max_hours, Nyears
    )

    attach_storageunits(n, costs, extendable_carriers, max_hours)
    attach_stores(n, costs, extendable_carriers)
    attach_hydrogen_pipelines(n, costs, extendable_carriers)
    attach_RCL_generators(
        n,
        snakemake.params["preinvestment_capacities"],
        snakemake.input["RCL_p_nom_limits"],
        snakemake.input["region_mapping"],
        snakemake.input["technology_cost_mapping"],
    )
    if snakemake.params["h2_demand"]["enabled"]:
        attach_hydrogen_demand(
            n,
            config=snakemake.params["h2_demand"],
            year=snakemake.wildcards["year"],
            fp_region_mapping=snakemake.input["region_mapping"],
            fp_remind_data=snakemake.input["remind_data"],
        )
    sanitize_carriers(n, snakemake.config)

    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    n.export_to_netcdf(snakemake.output[0])

# %%
