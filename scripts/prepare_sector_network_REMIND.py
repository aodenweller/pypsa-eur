# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2020-2024 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
"""
Adds the sector-coupling components to the network that are required for the
REMIND-PyPSA-Eur coupling. Currently this only includes EVs.
"""
# %%
import logging

import numpy as np
import pandas as pd
import pypsa
from _helpers import configure_logging, read_remind_data

logger = logging.getLogger(__name__)


# Function modified from prepare_sector_network.py
def add_EVs(n, ev_load, load_p_set, options_ev):

    # Estimate number of cars from EV load given assumptions in settings
    number_bev = (
        ev_load["value"]
        * options_ev["bev_share"]
        / options_ev["bev_annual_consumption"]
    )
    number_bet = (
        ev_load["value"]
        * (1 - options_ev["bev_share"])
        / options_ev["bet_annual_consumption"]
    )

    # Estimate simultaneous charging power in MW (used for link)
    charge_power = (
        number_bev * options_ev["bev_charge_rate"]
        + number_bet * options_ev["bet_charge_rate"]
    )

    # Estimate total battery pack capacity in MWh (used for store)
    battery_energy = (
        number_bev * options_ev["bev_energy"] + number_bet * options_ev["bet_energy"]
    )

    # Read in availability profile for charging
    link_avail_profile = pd.read_csv(
        snakemake.input.avail_profile, index_col=0, parse_dates=True
    )

    number_cars = pd.read_csv(snakemake.input.transport_data, index_col=0)[
        "number cars"
    ]
    # Distribute charg_power to spatial nodes using number of cars
    link_p_nom = charge_power.values[0] * number_cars / number_cars.sum()

    # Read in DSM profile
    store_dsm_profile = pd.read_csv(
        snakemake.input.dsm_profile, index_col=0, parse_dates=True
    )
    # Distribute DSM profile to spatial nodes using number of cars
    store_e_nom = battery_energy.values[0] * number_cars / number_cars.sum()

    n.add("Carrier", "EV battery")

    n.madd(
        "Bus",
        spatial_nodes,
        suffix=" EV battery",
        location=spatial_nodes,
        carrier="EV battery",
        unit="MWh_el",
    )

    p_shifted = (
        load_p_set + cycling_shift(load_p_set, 1) + cycling_shift(load_p_set, 2)
    ) / 3

    n.madd(
        "Load",
        spatial_nodes,
        suffix=" land transport EV",
        bus=spatial_nodes + " EV battery",
        carrier="land transport EV",
        p_set=p_shifted.loc[n.snapshots],  # TODO: Check why n.snapshots is necessary here
    )

    n.madd(
        "Link",
        spatial_nodes,
        suffix=" BEV charger",
        bus0=spatial_nodes,
        bus1=spatial_nodes + " EV battery",
        p_nom=link_p_nom,
        carrier="BEV charger",
        p_max_pu=link_avail_profile.loc[n.snapshots, spatial_nodes],
        # lifetime=1,
        efficiency=1,
    )

    # Exclude V2G for now
    # if options["v2g"]:
    #     n.madd(
    #         "Link",
    #         spatial_nodes,
    #         suffix=" V2G",
    #         bus1=spatial_nodes,
    #         bus0=spatial_nodes + " EV battery",
    #         p_nom=p_nom,
    #         carrier="V2G",
    #         p_max_pu=avail_profile[spatial_nodes],
    #         lifetime=1,
    #         efficiency=options.get("bev_charge_efficiency", 0.9),
    #     )

    if options_ev["dsm"]:

        n.madd(
            "Store",
            spatial_nodes,
            suffix=" EV battery",
            bus=spatial_nodes + " EV battery",
            carrier="EV battery",
            e_cyclic=True,
            e_nom=store_e_nom,
            e_max_pu=1,
            e_min_pu=store_dsm_profile.loc[n.snapshots, spatial_nodes],
        )


def cycling_shift(df, steps=1):
    """
    Cyclic shift on index of pd.Series|pd.DataFrame by number of steps.
    """
    df = df.copy()
    new_index = np.roll(df.index, steps)
    df.values[:] = df.reindex(index=new_index).values
    return df


def subtract_from_load(n, load_p_set):
    """
    Subtracts the EV load from the total load in the network.
    """
    n.loads_t["p_set"][spatial_nodes] -= load_p_set.loc[n.snapshots].values


# %%
if __name__ == "__main__":
    if "snakemake" not in globals():

        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "prepare_sector_network_REMIND",
            scenario="TEST",
            iteration="1",
            year="2040",
            simpl="",
            opts="3H-Ep134.4",
            clusters="4",
            ll="copt",
        )

    configure_logging(snakemake)

    # Read in EV parameters
    options_ev = snakemake.params.remind_settings["EVs"]

    # Read in network
    n = pypsa.Network(snakemake.input.network)

    # Get AC nodes from network
    spatial_nodes = n.buses.query("carrier == 'AC'").index

    # Read in electricity demand for EVs
    ev_load = read_remind_data(
        snakemake.input.remind_data,
        "p32_load_EVs",
        rename_columns={"ttot": "year", "all_regi": "region"},
    ).query("year == @snakemake.wildcards.year")
    ev_load["value"] *= 1e6 * 8760  # convert from TWa to MWh

    # Read in transport demand in units driven km [100 km]
    transport = pd.read_csv(
        snakemake.input.transport_demand, index_col=0, parse_dates=True
    )
    # Normalise such that the sum corresponds to the total EV electricity demand (in MWh)
    load_p_set = transport.div(transport.sum(axis=0).sum()) * ev_load["value"].values[0]
    
    # Add EVs to network
    add_EVs(n, ev_load, load_p_set, options_ev)

    # Subtract EV load from total load
    subtract_from_load(n, load_p_set)

    # Write network to file
    n.export_to_netcdf(snakemake.output.network)

# %%
