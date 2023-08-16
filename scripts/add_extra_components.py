# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2017-2023 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT

# coding: utf-8
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
from _helpers import configure_logging, get_region_mapping, get_technology_mapping
from add_electricity import load_costs, sanitize_carriers

idx = pd.IndexSlice

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


def attach_RCL_generators(n, fp_p_nom_limits, fp_region_mapping, fp_technology_mapping):
    """
    Add additional generators to network for the RCL constraint used in the
    REMIND-EU <-> PyPSA-EUR coupling.
    """

    p_nom_limits = pd.read_csv(fp_p_nom_limits)
    region_mapping = get_region_mapping(
        fp_region_mapping, source="REMIND-EU", target="PyPSA-Eur"
    )
    technology_mapping = get_technology_mapping(
        fp_technology_mapping, source="General", target="PyPSA-Eur"
    )

    # Apply mapping from REMIND/general to PyPSA-EUR countries and carriers
    p_nom_limits["country"] = p_nom_limits["region_REMIND"].map(region_mapping)
    p_nom_limits["carrier"] = p_nom_limits["general_technology"].map(technology_mapping)

    # Flatten country column entries such that all lists are converted into individual rows
    p_nom_limits = p_nom_limits.explode("country").explode("carrier")
    # Add country-reference to generators for mapping
    n.generators["country"] = n.generators["bus"].map(n.buses["country"])

    # Select all generators from n.generators where the combination of country and carrier can be found in p_nom_limits
    rcl_generators = n.generators.join(
        p_nom_limits.set_index(["country", "carrier"]),
        on=["country", "carrier"],
        how="right",
        rsuffix="_rcl",
    ).dropna(subset=["p_nom_max"])
    
    # Only consider RCL constraint for generators which are extendable
    rcl_generators = rcl_generators[rcl_generators["p_nom_extendable"] == True]

    # Modify properties of to-be-added RCL generators which differ from the original generators
    old_generators = rcl_generators.index
    rcl_generators.index = old_generators + " (RCL)"
    rcl_generators["capital_cost"] = 100 # small positive cost: should not be deployed for free for only few hours, but should definetly be deployed before regular capacities are built
    #rcl_generators["p_nom_extendable"] = True
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


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "add_extra_components",
            simpl="",
            clusters=4,
            scenario="test",
            iteration=1,
            year=2040,
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
        snakemake.input["RCL_p_nom_limits"],
        snakemake.input["region_mapping"],
        snakemake.input["technology_mapping"],
    )
    sanitize_carriers(n, snakemake.config)

    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    n.export_to_netcdf(snakemake.output[0])
