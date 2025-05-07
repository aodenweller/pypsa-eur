# -*- coding: utf-8 -*-
# %%
import logging
import os
import copy
import time
import gamspy as gt
import numpy as np
import pandas as pd
import pypsa
from _helpers import (
    configure_logging,
    get_region_mapping,
    get_technology_mapping,
    read_remind_data,
)
from scipy.stats import zscore

logger = logging.getLogger(__name__)

# ------------------------------
# Helper functions
# ------------------------------


def add_columns_for_processing(n, region_mapping, map_pypsaeur_to_general):
    """
    Add columns to network components:
    # (1) region (REMIND region)
    # (2) general_carrier (general technology group)
    # (3) RCL (region-carrier-limit) to identify the components more easily
    """
    # Remove columns from network components if they already exist
    # These come from the RCL constraints
    # TODO: Harmonise columns names in RCL constraints
    for comp in ["generators", "links", "stores"]:
        if "region_REMIND" in getattr(n, comp).columns:
            getattr(n, comp).drop(columns=["region_REMIND"], inplace=True)
        if "technology_group" in getattr(n, comp).columns:
            getattr(n, comp).drop(columns=["technology_group"], inplace=True)

    # (1) Add region
    # Add region to buses if it doesnt exist (this is the case if additional h2demand is not enabled)
    if "region" not in n.buses.columns:
        n.buses["region"] = ""

    # First map the PyPSA-EUR countries to REMIND-EU regions;
    # .statistics(..) can then automatically take care of the aggregation
    # H2 demand buses already have a region assigned, so we don't want to overwrite those
    n.buses["region"] = n.buses["region"].where(
        n.buses["region"] != "",
        n.buses["country"].map(region_mapping["REMIND-EU"]),
    )

    # HACK: Only temporary for one-region coupling!
    n.buses["region"] = "DEU"

    # Add information for aggregation later: region name (REMIND-EU) and general carrier
    n.generators["region"] = n.generators["bus"].map(n.buses["region"])
    n.stores["region"] = n.stores["bus"].map(n.buses["region"])
    n.storage_units["region"] = n.storage_units["bus"].map(n.buses["region"])
    n.links["region"] = n.links["bus0"].map(n.buses["region"])
    n.lines["region"] = n.lines["bus0"].map(n.buses["region"])
    n.loads["region"] = n.loads["bus"].map(n.buses["region"])
    # Links/lines have two buses, and can be attributed to two regions (used for e.g. grid length calculations)
    n.links["region1"] = n.links["bus1"].map(n.buses["region"])
    n.lines["region1"] = n.lines["bus1"].map(n.buses["region"])

    # (2) Add general_carrier
    n.generators["general_carrier"] = n.generators["carrier"].map(
        map_pypsaeur_to_general
    )
    n.stores["general_carrier"] = n.stores["carrier"]
    n.storage_units["general_carrier"] = n.storage_units["carrier"].map(
        map_pypsaeur_to_general
    )
    n.links["general_carrier"] = n.links["carrier"]
    n.lines["general_carrier"] = n.lines["carrier"]
    n.loads["general_carrier"] = n.loads["bus"].map(n.buses["carrier"])

    # (3) Add RCL
    n.generators["RCL"] = False
    n.generators.loc[n.generators.index.str.contains("RCL"), "RCL"] = True
    n.links["RCL"] = False
    n.links.loc[n.links.index.str.contains("RCL"), "RCL"] = True
    n.stores["RCL"] = False
    n.stores.loc[n.stores.index.str.contains("RCL"), "RCL"] = True


def get_pypsa_to_general_mapping(fp_mapping):
    """
    Get mapping from PyPSA-EUR to REMIND-EU technologies.
    """
    map_pypsaeur_to_general = (
        get_technology_mapping(fp_mapping, group_technologies=True)
        .groupby("PyPSA-Eur")
        .agg(lambda x: list(set(x))[0])["technology_group"]
        .to_dict()
    )
    map_pypsaeur_to_general.pop("offwind")  # not needed
    # Add some link mappings manually
    # TODO: Clean up (move to mapping file?)
    map_pypsaeur_to_general["H2 electrolysis"] = "electrolysis"
    map_pypsaeur_to_general["H2 fuel cell"] = "fuel cell"
    map_pypsaeur_to_general["battery charger"] = "battery charger"
    map_pypsaeur_to_general["battery discharger"] = "battery discharger"
    # Add store mappings manually
    # TODO: Clean up (move to mapping file?)
    map_pypsaeur_to_general["H2"] = "hydrogen storage underground"
    map_pypsaeur_to_general["battery"] = "battery storage"

    return map_pypsaeur_to_general


def get_general_to_remind_mapping(fp_mapping):
    """
    Get mapping from general technologies to REMIND-EU technologies.
    """
    map_general_to_remind = (
        get_technology_mapping(fp_mapping, group_technologies=True)
        .groupby("technology_group")
        .agg(lambda x: list(set(x)))["REMIND-EU"]
        .to_dict()
    )
    # Add some link mappings manually
    # Battery charger and discharger need to be added manually because
    # REMIND2PyPSA cost input goes via "battery inverter", but
    # PyPSA2REMIND goes via "battery charger" and "battery discharger"
    # TODO: Clean up (move to mapping file?)
    map_general_to_remind["battery charger"] = ["btin"]
    map_general_to_remind["battery discharger"] = ["btout"]
    map_general_to_remind["H2 electrolysis"] = ["elh2"]
    map_general_to_remind["H2 fuel cell"] = ["h2turb"]
    map_general_to_remind["H2"] = ["h2stor"]
    map_general_to_remind["battery"] = ["btstor"]

    return map_general_to_remind


def get_pypsa_to_remind_region_mapping(fp_region_mapping):
    """
    Get mapping from PyPSA-EUR to REMIND-EU regions.
    """
    region_mapping = get_region_mapping(
        fp_region_mapping, source="PyPSA-EUR", target="REMIND-EU"
    )
    region_mapping = pd.DataFrame(region_mapping).T.reset_index()
    region_mapping.columns = ["PyPSA-EUR", "REMIND-EU"]
    region_mapping = region_mapping.set_index("PyPSA-EUR")

    return region_mapping


def check_for_mapping_completeness(n):
    """
    Check if all carriers in the network have been mapped to general technologies and
    if all general technologies have been mapped to REMIND-EU technologies.
    """
    if (
        tmp_set := set(n.generators["carrier"])
        - map_pypsaeur_to_general.keys()
        - {"load"}
    ):
        logger.info(
            f"Technologies (carriers) missing from mapping PyPSA-EUR -> general technologies:\n {tmp_set}"
        )

    if tmp_set := map_pypsaeur_to_general.values() - map_general_to_remind.keys():
        logger.info(
            f"Technologies (carriers) missing from mapping General -> REMIND-EU:\n {tmp_set}"
        )

    if tmp_set := set(n.loads["general_carrier"]) - map_pypsaeur_to_remind_loads.keys():
        logger.info(
            f"Technologies (carriers) missing from mapping PyPSA-EUR -> REMIND-EU (loads):\n {tmp_set}"
        )

    if tmp_set := set(n.buses["country"]) - set(region_mapping.index):
        logger.info(
            f"PyPSA-EUR countries without mapping to REMIND-EU regions::\n {tmp_set}"
        )


def process_data(df, cols, map_to_remind):
    """
    Process the dataframes, combines the network-specific results into one dataframe
    removes excess columns / sets index and sorts by cols.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to process.
    cols : list
        Columns to keep in the resulting dataframe.
    map_to_remind: bool
        Whether to map the general carrier names to REMIND carrier names.
    """
    # Set index dynamically and sort
    df = df.set_index(cols).sort_index()

    # Reset index
    df = df.reset_index().drop(columns=["level_0"], errors="ignore")

    # Remove rows related to the additional hydrogen bus
    if "general_carrier" in df.columns:
        df = df.query(
            "general_carrier not in ['H2 transfer to H2 demand REMIND', 'H2 demand buffer REMIND']"
        )

    # Remove rows related to DC if available
    if "general_carrier" in df.columns:
        df = df.query("general_carrier != 'DC'")

    # Remove rows related to EVs if available
    if "general_carrier" in df.columns:
        df = df.query("general_carrier != 'BEV charger'")
        df = df.query("general_carrier != 'EV battery'")

    # Function to map and explode carrier columns
    def map_and_explode(df, column):
        if column in df.columns:
            # Get original column order
            cols = list(df.columns)
            # Map carriers to REMIND technologies
            df.loc[:, f"new_{column}"] = df[column].map(map_general_to_remind)
            # Explode the new carriers into separate rows
            df = (
                df.explode(f"new_{column}")
                .drop(columns=[column])
                .rename(columns={f"new_{column}": column})
            )
            # Reorder columns
            df = df[cols]
        return df

    # Map carriers to REMIND technologies
    if map_to_remind:
        df = map_and_explode(
            df.copy(), "general_carrier"
        )  # Make a copy before modifying
        if "carrier_perturbed" in df.columns:
            df = map_and_explode(
                df.copy(), "carrier_perturbed"
            )  # Make a copy before modifying

    return df


# Helper function to weigh data by REMIND capacities for n:m mappings
def weigh_by_REMIND_capacity(df, grouper, year):
    """
    Weighs data using REMIND capacities, ensuring weights sum to 1 per general carrier group.
    """
    # Load and preprocess REMIND capacity weights
    capacity_weights = (
        read_remind_data(
            file_path=snakemake.input["remind_weights"],
            variable_name="p32_weightGen",
            rename_columns={"ttot": "year", "all_regi": "region", "all_te": "carrier"},
        )
        .astype({"year": int, "value": float, "carrier": str, "region": str})
        .query(f"year == {year}")
    )

    storage_weights = (
        read_remind_data(
            file_path=snakemake.input["remind_weights"],
            variable_name="p32_weightStor",
            rename_columns={"ttot": "year", "all_regi": "region", "all_te": "carrier"},
        )
        .astype({"year": int, "value": float, "carrier": str, "region": str})
        .query(f"year == {year}")
    )

    # HACK: Only temporary until output from REMIND resolved
    # Add dummy carriers for btin, btout, h2stor, btstor
    storage_weights = pd.concat(
        [
            storage_weights,
            pd.DataFrame(
                {
                    "year": [year] * 4,
                    "region": ["DEU"] * 4,
                    "carrier": ["btin", "btout", "h2stor", "btstor"],
                    "value": [1.0] * 4,
                }
            ),
        ]
    )
    weights = pd.concat([capacity_weights, storage_weights])

    # Remove near-zero values
    weights["value"] = weights["value"].where(
        weights["value"] > np.finfo(float).eps, 0.0
    )

    # Map carriers to general carriers
    weights["general_carrier"] = weights["carrier"].map(
        {lv: k for k, v in map_general_to_remind.items() for lv in v}
    )

    # Compute total levels per general carrier
    general_carrier_weights = (
        weights.groupby(["year", "region", "general_carrier"])["value"]
        .sum()
        .replace(0, 1)
    )  # Avoid division by zero

    # Compute individual weights
    weights = weights.join(
        general_carrier_weights.rename("general_carrier_weight"),
        on=["year", "region", "general_carrier"],
    )
    weights["weight"] = weights["value"] / weights["general_carrier_weight"]
    weights.drop(columns=["value"], inplace=True)

    # Apply weights to data
    df = df.merge(weights, on=grouper, how="left")
    assert df["weight"].notna().all(), "Some weights are missing"
    assert df["weight"].between(0.0, 1.0).all(), "Invalid weight values"

    df["value"] *= df["weight"]
    return df[["region", "carrier", "value"]]


# ------------------------------
# Coupling functions
# ------------------------------
def calculate_capacity_factors(n, comps, grouper, map_to_remind):
    """
    Calculate capacity factors for components in the network.

    Parameters
    ----------
    n : pypsa.Network
        Network to calculate capacity factors for.
    comps : list
        List of components to calculate capacity factors for.
    grouper : list
        List of columns to group the capacity factors by.
    map_to_remind: bool
        Whether to map the general carrier names to REMIND carrier names.

    Returns
    -------
    pd.DataFrame
        Capacity factors for the components in the network.
    """
    # Calculate capacity factors
    capacity_factors = n.statistics.capacity_factor(comps=comps, groupby=grouper)
    capacity_factors = (
        capacity_factors.to_frame("value").reset_index().drop(columns=["component"])
    )

    return process_data(capacity_factors, cols=grouper, map_to_remind=map_to_remind)


def calculate_load_prices(n, grouper):
    """
    Calculate load prices for the network.

    Parameters
    ----------
    n : pypsa.Network
        Network to calculate load prices for.
    grouper : list
        List of columns to group the load prices by.
    """
    # Calculate load prices
    load_prices = n.statistics.revenue(comps=["Load"], groupby=grouper) / (
        -1 * n.statistics.withdrawal(comps=["Load"], groupby=grouper)
    )
    load_prices = (
        load_prices.to_frame("value").reset_index().drop(columns=["component"])
    )

    return load_prices


def calculate_load_prices_with_cutoff(n, grouper, z_cutoff):
    """
    Calculate load prices for the network with a z-score cutoff.

    Parameters
    ----------
    n : pypsa.Network
        Network to calculate load prices for.
    grouper : list
        List of columns to group the load prices by.
    z_cutoff: float
        Z-score above which to cut off scarcity prices.
    """
    # Cutoff scarcity prices if configured
    if z_cutoff:
        n_calc = cutoff_scarcity_prices(n, z_cutoff)
    else:
        n_calc = n

    # Calculate load prices
    load_prices = calculate_load_prices(n_calc, grouper)

    return load_prices


def calculate_markups_supply(n, comps, grouper, z_cutoff, map_to_remind):
    """
    Calculate markups for all generators.

    Parameters
    ----------
    n : pypsa.Network
        Network to calculate markups for.
    comps : list
        List of components to calculate markups for.
    grouper : list
        List of columns to group the markups by.
    z_cutoff: float or bool
        Z-score above which to cut off scarcity prices.
    map_to_remind: bool
        Whether to map the general carrier names to REMIND carrier names.
    """
    # Cutoff scarcity prices if configured
    if z_cutoff:
        n_calc = cutoff_scarcity_prices(n, z_cutoff)
    else:
        n_calc = n

    # Calculate markups for the supply side
    market_value = n_calc.statistics.market_value(comps=comps, groupby=grouper)

    # Get average electricity price
    load_price_ac = (
        calculate_load_prices(n_calc, grouper).query("general_carrier == 'AC'").value[0]
    )

    # Subtract average electricity price from market value to get markup
    markups_supply = market_value - load_price_ac
    markups_supply = (
        markups_supply.to_frame("value")
        .reset_index()
        .drop(columns=["component"])
        .fillna(0)
    )

    return process_data(markups_supply, cols=grouper, map_to_remind=map_to_remind)


# TODO: Make compatible with multiple regions
def calculate_markups_demand(n, grouper, z_cutoff, map_to_remind):
    """
    Calculate markups for the demand side, i.e.
    electricity prices paid by differend end-users.
    Currently only implemented for electrolysis.

    Parameters
    ----------
    n : pypsa.Network
        Network to calculate markups for.
    grouper : list
        List of columns to group the markups by.
    z_cutoff: float or bool
        Z-score above which to cut off scarcity prices.
    map_to_remind: bool
        Whether to map the general carrier names to REMIND carrier names.
    """
    # Cutoff scarcity prices if configured
    if z_cutoff:
        n_calc = cutoff_scarcity_prices(n, z_cutoff)
    else:
        n_calc = n

    # Electrolysis generation series
    gen = n_calc.links_t.p0.loc[:, n_calc.links.carrier == "H2 electrolysis"]
    gen.columns = gen.columns.map(n_calc.links.bus0)
    gen = gen.T.groupby(level=0).sum().T
    # Local marginal price
    lmp = n_calc.buses_t.marginal_price.loc[:, gen.columns]
    # Calculate market value
    mv = (gen * lmp).sum().sum() / gen.sum().sum()

    # Get average electricity price
    load_price_ac = (
        calculate_load_prices(n_calc, grouper).query("general_carrier == 'AC'").value[0]
    )

    # Subtract average electricity price from market value to get markup
    markup_electrolysis = mv - load_price_ac

    # Calculate markups for EV charging
    gen = n_calc.links_t.p0.loc[:, n_calc.links.carrier == "BEV charger"]
    gen.columns = gen.columns.map(n_calc.links.bus0)
    gen = gen.T.groupby(level=0).sum().T
    # Local marginal price
    lmp = n_calc.buses_t.marginal_price.loc[:, gen.columns]
    # Calculate market value
    mv = (gen * lmp).sum().sum() / gen.sum().sum()

    # Subtract average electricity price from market value to get markup
    markup_evs = mv - load_price_ac
    
    # Replace NA with 0
    markup_electrolysis = markup_electrolysis if not np.isnan(markup_electrolysis) else 0
    markup_evs = markup_evs if not np.isnan(markup_evs) else 0

    # Create DataFrame
    markups_demand = pd.DataFrame(
        [
            {
                "region": "DEU",
                "general_carrier": "electrolysis",
                "value": markup_electrolysis,
            },
            {
                "region": "DEU",
                "general_carrier": "EVs",
                "value": markup_evs,
            },
        ]
    )

    return process_data(markups_demand, cols=grouper, map_to_remind=map_to_remind)


def calculate_peak_residual_loads(n, grouper, kind):
    """
    Calculate peak residual loads for the network.

    Parameters
    ----------
    n : pypsa.Network
        Network to calculate peak residual loads for.
    kind: str
        Kind of peak residual load to calculate. Can be "absolute", "relative" or ["absolute", "relative"].
    """
    ## Calculate peak residual load
    dispatchable_technologies = set(n.generators.index) - set(
        n.generators_t.p_max_pu.columns
    )
    # Add attribute to network.generators to distinguish between dispatchable and non-dispatchable technologies
    n.generators["peak_residual_load"] = "No"
    n.generators.loc[list(dispatchable_technologies), "peak_residual_load"] = "Yes"
    # Don't include load shedding as dispatchable technology
    n.generators.loc[n.generators.index.str.contains("load"), "peak_residual_load"] = (
        "No"
    )
    n.loads["peak_residual_load"] = "Load"
    # Don't include hydrogen turbines and batteries into peak residual load calculation
    n.stores["peak_residual_load"] = "No"
    # Don't include hydro and pumped hydro into peak residual load calculation (no PHS in REMIND)
    n.storage_units["peak_residual_load"] = "No"

    residual_load = (
        n.statistics.energy_balance(
            comps=["Generator", "Store", "StorageUnit", "Load"],
            bus_carrier="AC",
            groupby=[grouper, "peak_residual_load"],
            aggregate_time=False,
        )
        .groupby([grouper, "peak_residual_load"])
        .sum()
    )

    # Helper function to be used with groupby
    def get_absolute_and_relative_prl(x):
        # Find the snapshot with absolute peak residual load
        max_prl_snapshot = x.xs("Yes", level="peak_residual_load").idxmax(
            axis="columns"
        )
        return pd.Series(
            {
                # Use snapshot to determine absolute and calculate relative peak residual load
                "absolute": x.xs("Yes", level="peak_residual_load")[max_prl_snapshot]
                .iloc[0]
                .item(),
                # relative means relative to the average load (given by v32_load)
                "relative": (
                    x.xs("Yes", level="peak_residual_load")[max_prl_snapshot]
                    .iloc[0]
                    .item()
                    / (
                        -1 * x.xs("Load", level="peak_residual_load").mean(axis=1)
                    ).item()
                ),
            }
        )

    peak_residual_load = (
        residual_load.groupby(grouper)
        .apply(get_absolute_and_relative_prl)
        .reset_index()
    )

    # Select type
    peak_residual_load = peak_residual_load[[grouper, kind]]

    return peak_residual_load


def calculate_availability_factors(n, comps, grouper, map_to_remind):
    """
    Calculate the availability factor of generators in the network.

    Only public PyPSA API is used (future-proof).
    """

    # 1. Compute availability time-series
    p_max_pu = n.generators_t.p_max_pu
    p_nom_opt = n.generators.p_nom_opt

    availability = p_max_pu.multiply(p_nom_opt, axis=1)

    # 2. Get snapshot weightings (important if snapshots are not equally spaced)
    snapshot_weightings = n.snapshot_weightings.generators

    # 3. Aggregate over time using weighting
    weighted_availability = (availability.T @ snapshot_weightings) / snapshot_weightings.sum()

    # 4. Attach generator attributes (like carrier, technology) to availability
    weighted_availability = weighted_availability.rename_axis('Generator').to_frame('availability')
    weighted_availability = weighted_availability.join(n.generators[grouper])

    # 5. Group by desired category (e.g., carrier, bus, technology) and sum
    df = weighted_availability.groupby(grouper).sum()

    # 6. Now divide by total capacity
    total_capacity = n.generators.groupby(grouper).p_nom_opt.sum()

    df['value'] = df['availability'] / total_capacity
    
    df = df.drop(columns=['availability']).reset_index()

    return process_data(df, cols=grouper, map_to_remind=map_to_remind)


def calculate_potentials(n, grouper, map_to_remind):
    """
    Calculate VRE potentials for the network.

    Parameters
    ----------
    n : pypsa.Network
        Network to calculate potentials for.
    grouper : list
        List of columns to group the potentials by.
    map_to_remind: bool
        Whether to map the general carrier names to REMIND carrier names.
    """
    # RCL generators have to be excluded from potentials
    df = n.generators.copy(deep=True)
    df = df.query("not index.str.contains('RCL')", engine="python")
    potential = df.groupby(grouper)["p_nom_max"].sum()
    potential = potential.replace([np.inf, -np.inf], np.nan).dropna()
    potential = potential.to_frame("value").reset_index()

    return process_data(potential, cols=grouper, map_to_remind=map_to_remind)


def calculate_optimal_capacities(n, comps, grouper, weigh_by_remind, year=None):
    """
    Calculate optimal capacities for the network.

    Parameters
    ----------
    n : pypsa.Network
        Network to calculate optimal capacities for.
    comps : list
        List of components to calculate optimal capacities for.
    grouper : list
        List of columns to group the optimal capacities by.
    weigh_by_remind: bool
        Whether to weigh the optimal capacities by REMIND capacities.
    year: int
        Year to weigh the optimal capacities by REMIND capacities.
    """
    # Calculate optimal capacities
    optimal_capacities = n.statistics.optimal_capacity(comps=comps, groupby=grouper)
    optimal_capacities = optimal_capacities.to_frame("value").reset_index()

    # Remove rows related to the additional hydrogen bus
    if "general_carrier" in optimal_capacities.columns:
        optimal_capacities = optimal_capacities.query(
            "general_carrier != 'H2 transfer to H2 demand REMIND'"
        )
        optimal_capacities = optimal_capacities.query(
            "general_carrier != 'H2 demand buffer REMIND'"
        )

    # Weigh by REMIND capacities
    if weigh_by_remind:
        # Drop
        optimal_capacities = optimal_capacities.drop(columns=["component"])
        # Drop DC link for now
        optimal_capacities = optimal_capacities.query("general_carrier != 'DC'")
        # Drop EV battery links for now
        optimal_capacities = optimal_capacities.query(
            "general_carrier != 'BEV charger'"
        )
        # Drop EV batteries for now
        optimal_capacities = optimal_capacities.query(
            "general_carrier != 'EV battery'"
        )
        # Ensure year is provided
        if year is None:
            raise ValueError("Year must be provided to weigh by REMIND capacities")
        optimal_capacities = weigh_by_REMIND_capacity(optimal_capacities, grouper, year)

    return optimal_capacities


def calculate_grid_losses(n, grouper="region", kind="relative"):
    """
    Calculate grid losses for the network.

    Parameters
    ----------
    n : pypsa.Network
        Network to calculate grid losses for.
    kind: str or list
        Kind of grid losses to calculate. Can be "absolute" or "relative" or ["absolute", "relative"].
    """
    ## Determine grid losses in absolute and relative terms
    grid_loss_abs = n.statistics.energy_balance(comps="Line", groupby=grouper).abs()
    # Handle the case where there are no grid losses (e.g., no lines)
    if grid_loss_abs.empty:
        regions = n.buses["region"].unique()
        grid_loss_abs = pd.Series(0, index=regions, name="absolute")
        grid_loss_abs.index.name = "region"
    grid_loss_rel = grid_loss_abs / n.statistics.withdrawal(
        comps="Load", bus_carrier = ["AC", "EV battery"], groupby=grouper
    )

    grid_loss = pd.DataFrame(
        {"absolute": grid_loss_abs, "relative": grid_loss_rel}
    ).reset_index()

    grid_loss = grid_loss[[grouper, kind]]

    return grid_loss


def calculate_link_generation(n, carrier, grouper, kind="relative"):
    """
    Calculate relative or absolute generation of link for the network.

    Parameters
    ----------
    n : pypsa.Network
        Network to calculate link generation for.
    carrier : str
        Carrier to calculate link generation for.
    kind: str or list
        Kind of link generation to calculate. Can be "absolute" or "relative" or ["absolute", "relative"].
    """

    def get_supply_with_zeros(n, carrier, component="Link"):
        # Extract supply data for the specified component
        supply_data = n.statistics.supply(
            comps=[component], groupby=["region", "carrier"]
        )
        supply_data = supply_data.xs(
            component, level="component"
        )  # Filter by component

        # Convert to DataFrame, unstack "carrier", and fill missing values with zeros
        supply_df = supply_data.unstack(level="carrier").fillna(0)

        # Ensure the specified carrier column exists
        if carrier not in supply_df.columns:
            supply_df[carrier] = 0

        # Extract the supply for the specified carrier
        supply_series = supply_df[carrier]

        # Add "component" level to the index and reorder levels
        supply_series = supply_series.to_frame("objective")
        supply_series["component"] = component
        supply_series = supply_series.set_index("component", append=True)
        supply_series = supply_series.reorder_levels(["component", "region"])[
            "objective"
        ]

        return supply_series

    # Get absolute supply for the specified carrier
    # TODO: Use parameter "grouper"
    absolute_supply = get_supply_with_zeros(n, carrier=carrier, component="Link")

    # Calculate relative supply
    relative_supply = absolute_supply / n.statistics.withdrawal(
        comps="Load", bus_carrier = ["AC", "EV battery"], groupby=grouper
    )

    # Combine absolute and relative supply into a DataFrame
    link_generation = (
        pd.DataFrame(
            {
                "absolute": absolute_supply,
                "relative": relative_supply.fillna(0),  # Fill NaN values with 0
            }
        )
        .reset_index()
        .drop(columns=["component"])
    )

    link_generation = link_generation[[grouper, kind]]

    return link_generation


def calculate_generation_shares(n, grouper, year):
    """
    Calculate generation shares for the network.

    Parameters
    ----------
    n : pypsa.Network
        Network to calculate generation shares for.
    grouper : list
        List of columns to group the generation shares by.
    year : int
        Year to calculate generation shares for.
    """
    # Calculate shares of technologies in annual generation
    generation_share = (
        n.statistics.supply(
            comps=["Generator"],
            groupby=grouper,
        )
        / n.statistics.supply(
            comps=["Generator"],
            groupby=grouper,
        )
        .groupby(["region"])
        .sum()
    )
    generation_shares = (
        generation_share.to_frame("value").reset_index().drop(columns=["component"])
    )

    generation_shares = weigh_by_REMIND_capacity(generation_shares, grouper, year)

    return generation_shares


def calculate_difference_quotient(
    n_opt, n_pert, ptech, property_func, grouper, exclude=None, **kwargs
):
    """
    Calculate the difference quotient as a numerical approximation of the partial
    derivative of a property function with respect to the capacity of a specific technology.

    Parameters
    ----------
    n_opt : pypsa.Network
        Optimal PyPSA network.
    n_pert : pypsa.Network
        Perturbed PyPSA network.
    ptech : str
        Technology that was perturbed.
    property_func : function
        Function for which the difference quotient is calculated.
    grouper : list
        List of columns to group the results by.
    exclude: list
        List of general_carrier elements to exclude from the results.
    """

    def get_filtered_capacity(n, ptech, grouper):
        # Get optimal capacity of the perturbed technology
        return (
            n.statistics.optimal_capacity(comps="Generator", groupby=grouper)
            .to_frame()
            .reset_index()
            .loc[
                lambda x: x["general_carrier"].str.contains(ptech, case=False, na=False)
            ]
        )

    # Get the property values for optimal and perturbed networks
    prop_opt = property_func(n_opt, grouper=grouper, **kwargs)
    prop_pert = property_func(n_pert, grouper=grouper, **kwargs)
    prop_merged = prop_opt.merge(prop_pert, on=grouper, suffixes=("_orig", "_pert"))

    # Extract total capacities for the perturbed technology
    cap_orig = get_filtered_capacity(n_opt, ptech, grouper)
    cap_pert = get_filtered_capacity(n_pert, ptech, grouper)
    cap_merged = cap_orig.merge(cap_pert, on=grouper, suffixes=("_orig", "_pert"))
    cap_merged.rename(columns={"general_carrier": "carrier_perturbed"}, inplace=True)

    # Merge capacity with property values
    merged = prop_merged.merge(cap_merged, on="region")

    # Exclude unwanted components from both general_carrier and carrier_perturbed
    if exclude:
        merged = merged.query(
            "general_carrier not in @exclude and carrier_perturbed not in @exclude"
        ).copy()

    # Compute difference quotient
    merged["value"] = (merged["value_pert"] - merged["value_orig"]) / (
        merged["p_nom_opt_pert"] - merged["p_nom_opt_orig"]
    )
    merged = merged[["region", "general_carrier", "carrier_perturbed", "value"]]
    merged = process_data(merged, cols=grouper, map_to_remind=True)

    return merged


# Currently not in use
def determine_crossborder_flow_and_price(network, carrier=["AC", "DC"]):
    """
    Function to determine (i) electricity exports by lines and links
    and (ii) corresponding electricity prices paid by the importing
    region (i.e. paid by region in "to" to region in "from").

    Restricted to carrier. Aggregated to REMIND regions.
    Returns annual aggregates of (i) and (ii).
    """

    # Determine relevant connector between regions
    relevant_connectors = pd.concat([network.links, network.lines]).query(
        "carrier in @carrier and region!=region1"
    )[["bus0", "bus1"]]

    # Read both p0 and p1 of both links and lines
    p0 = pd.concat([network.links_t["p0"], network.lines_t["p0"]], axis="columns")[
        relevant_connectors.index
    ]
    p1 = pd.concat([network.links_t["p1"], network.lines_t["p1"]], axis="columns")[
        relevant_connectors.index
    ]

    # Map relevant_connectors to buses, from which marginal prices are taken
    p0.columns = pd.MultiIndex.from_frame(
        relevant_connectors.loc[p0.columns, ["bus0", "bus1"]],
        names=["from", "to"],
    )
    p1.columns = pd.MultiIndex.from_frame(
        relevant_connectors.loc[
            p1.columns, ["bus1", "bus0"]
        ],  # Reverse order as p1 is the reverse flow
        names=["from", "to"],
    )

    # Concatenate both and filter for positive values (exports)
    # This is fine because we have included both p0 and p1
    p = pd.concat([p0, p1], axis="columns").where(lambda x: x > 0)

    # Apply snapshot weightings if the time resolution is not hourly (1H)
    p = p.mul(network.snapshot_weightings["objective"], axis="rows")

    # Get marginal prices at importing buses, i.e. at "to" buses
    price_import = network.buses_t["marginal_price"][p.columns.get_level_values("to")]
    price_import.columns = p.columns

    # Get marginal prices at exporting buses, i.e. at "from" buses
    price_export = network.buses_t["marginal_price"][p.columns.get_level_values("from")]
    price_export.columns = p.columns

    # Calculate total expenses for importing and revenue for exporting
    expense_import = p.mul(price_import).T
    revenue_export = p.mul(price_export).T

    # Map buses to regions
    expense_import.index = pd.MultiIndex.from_arrays(
        [
            expense_import.index.get_level_values("from").map(network.buses["region"]),
            expense_import.index.get_level_values("to").map(network.buses["region"]),
        ],
        names=["from", "to"],
    )

    revenue_export.index = pd.MultiIndex.from_arrays(
        [
            revenue_export.index.get_level_values("from").map(network.buses["region"]),
            revenue_export.index.get_level_values("to").map(network.buses["region"]),
        ],
        names=["from", "to"],
    )

    # Sum over hours
    expense_import = expense_import.groupby(["from", "to"]).sum().sum(axis="columns")
    revenue_export = revenue_export.groupby(["from", "to"]).sum().sum(axis="columns")

    # Transpose
    p = p.T

    # Map buses to regions
    p.index = pd.MultiIndex.from_arrays(
        [
            p.index.get_level_values("from").map(network.buses["region"]),
            p.index.get_level_values("to").map(network.buses["region"]),
        ],
        names=["from", "to"],
    )

    # Calculate total electricity flow
    p = p.groupby(["from", "to"]).sum().sum(axis="columns")

    # Calculate average electricity price paid by importing region in EUR/MWh
    price_import_avg = expense_import / p
    price_export_avg = revenue_export / p

    # If value in p is zero (no crossborder in entire year), replace NaN price with 1
    price_import_avg = price_import_avg.where(p > 0, 1)
    price_export_avg = price_export_avg.where(p > 0, 1)

    # Convert to dataframe
    p = p.to_frame("exports").reset_index()
    price_import_avg = price_import_avg.to_frame("price").reset_index()
    price_export_avg = price_export_avg.to_frame("price").reset_index()

    return p, price_import_avg, price_export_avg


# ------------------------------
# Reporting functions
# ------------------------------


def calculate_energy_balance(n, grouper, bus_carrier, comps=None):
    """
    Calculate the energy balance for the network.

    Parameters
    ----------
    n : pypsa.Network
        Network to calculate the energy balance for.
    """
    # Calculate energy balance
    energy_balance = (
        n.statistics.energy_balance(
            comps=comps, groupby=grouper, bus_carrier=bus_carrier, nice_names=False
        )
        .to_frame("value")
        .reset_index()
    )

    return energy_balance


def calculate_preinstalled_capacities(n, grouper):
    """
    Calculate preinstalled capacities for the network.
    These are capacities passed for free from REMIND to PyPSA.

    Parameters
    ----------
    n : pypsa.Network
        Network to calculate preinstalled capacities for.
    """
    # Preinstalled capacities consist of two parts
    # First, get p_nom from capacity-adjusted existing powerplants
    preinstalled_capacity_ppl = n.statistics.installed_capacity(
        comps=["Generator"], groupby=(["RCL"] + grouper)  # Only generators for now
    )
    preinstalled_capacity_ppl = preinstalled_capacity_ppl.to_frame(
        "value"
    ).reset_index()
    preinstalled_capacity_ppl = preinstalled_capacity_ppl.query("RCL == False")
    # Second, get p_nom_opt from RCL components (also for links and stores)
    preinstalled_capacity_rcl = n.statistics.optimal_capacity(
        comps=["Generator", "Link", "Store"], groupby=(["RCL"] + grouper)
    )
    preinstalled_capacity_rcl = preinstalled_capacity_rcl.to_frame(
        "value"
    ).reset_index()
    preinstalled_capacity_rcl = preinstalled_capacity_rcl.query("RCL == True")
    # Combine both
    preinstalled_capacities = pd.concat(
        [preinstalled_capacity_ppl, preinstalled_capacity_rcl]
    )

    return preinstalled_capacities


def cutoff_scarcity_prices(n, z_cutoff):
    """
    Create a deep copy of the network and apply a
    cutoff for scarcity prices to the marginal prices.

    Parameters
    ----------
    n : pypsa.Network
        Network to apply cutoff for scarcity prices to.
    z_cutoff : float
        Z-score to apply cutoff for scarcity prices.
    """
    # Calculate market values after applying cutoff for electricity prices
    n_cutoff = copy.deepcopy(n)

    relevant_buses = n_cutoff.buses.query("carrier == 'AC'").index
    z_cutoff = float(z_cutoff)
    zscores = (
        n_cutoff.buses_t["marginal_price"][relevant_buses]
        .apply(zscore)
        .mean(axis="columns")
    )
    zscores.index = pd.to_datetime(zscores.index)

    # By setting snapshot_weightings to 0, the market value will not be calculated for these snapshots above the cutoff value
    n_cutoff.snapshot_weightings = n_cutoff.snapshot_weightings.where(
        zscores < z_cutoff, 0
    )

    logger.info(
        "Excluding {number} snapshots from calculations with electricity prices above {p:.2f} USD/MWh.".format(
            number=int(
                n_cutoff.snapshot_weightings["generators"].shape[0]
                * n_cutoff.snapshot_weightings["generators"].iloc[0]
                - n_cutoff.snapshot_weightings["generators"].sum()
            ),
            p=n_cutoff.buses_t["marginal_price"][relevant_buses]
            .where(zscores < z_cutoff)
            .mean(axis="columns")
            .max(),
        )
    )

    return n_cutoff


def calculate_market_values_supply(n, comps, grouper, z_cutoff):
    """
    Calculate market values for

    Parameters
    ----------
    n : pypsa.Network
        Network to calculate market values for.
    comps: list
        List of components to calculate market values for.
    grouper : list
        List of columns to group the market values by.
    z_cutoff: float
        Z-score above which to cut off scarcity prices.
    """
    # Cutoff scarcity prices if configured
    if z_cutoff:
        n_calc = cutoff_scarcity_prices(n, z_cutoff)
    else:
        n_calc = n

    # Calculate the market values (round-about way as the intended method of the statistics module is not yet available)
    market_values_supply = n_calc.statistics.market_value(
        comps=comps,
        groupby=grouper,
    )
    market_values_supply = market_values_supply.to_frame("value").reset_index()

    return market_values_supply


def calculate_market_values_demand(n, z_cutoff):
    """
    Calculate market values for demand-side end-users.

    Parameters
    ----------
    n : pypsa.Network
        Network to calculate market values for.
    grouper : list
        List of columns to group the market values by.
    z_cutoff: float
        Z-score above which to cut off scarcity prices.
    """
    # Cutoff scarcity prices if configured
    if z_cutoff:
        n_calc = cutoff_scarcity_prices(n, z_cutoff)
    else:
        n_calc = n

    # Electrolysis generation series
    gen = n_calc.links_t.p0.loc[:, n_calc.links.carrier == "H2 electrolysis"]
    gen.columns = gen.columns.map(n_calc.links.bus0)
    gen = gen.T.groupby(level=0).sum().T
    # Local marginal price
    lmp = n_calc.buses_t.marginal_price.loc[:, gen.columns]
    # Calculate market value
    mv_electrolysis = (gen * lmp).sum().sum() / gen.sum().sum()

    # EV load series
    # TODO: This is the same as the load price at the EV battery bus, simplify
    gen = n_calc.loads_t.p.loc[:, n_calc.loads.carrier == "land transport EV"]
    gen.columns = gen.columns.map(n_calc.loads.bus)
    gen = gen.T.groupby(level=0).sum().T
    # Local marginal price
    lmp = n_calc.buses_t.marginal_price.loc[:, gen.columns]
    # Calculate market value
    mv_evs = (gen * lmp).sum().sum() / gen.sum().sum()

    # Create DataFrame
    market_values_demand = pd.DataFrame(
        [
            {
                "region": "DEU",
                "general_carrier": "electrolysis",
                "value": mv_electrolysis,
            },
            {
                "region": "DEU",
                "general_carrier": "EVs",
                "value": mv_evs,
            },
        ]
    )

    return market_values_demand


def calculate_curtailments(n, grouper):
    """
    Calculate curtailments for the network.

    Parameters
    ----------
    n : pypsa.Network
        Network to calculate curtailments for.
    grouper : list
        List of columns to group the curtailments by.
    """
    # Calculate curtailments
    curtailments = n.statistics.curtailment(groupby=grouper, nice_names=False)
    curtailments = curtailments.to_frame("value").reset_index()

    return curtailments


# TODO: Check implementation and compare with other options
def calculate_hourly_prices(n, grouper):
    """
    Calculate hourly prices for the network.

    Parameters
    ----------
    n : pypsa.Network
        Network to calculate hourly prices for.
    grouper : list
        List of columns to group the hourly prices by.
    """
    # Calculate hourly prices
    hourly_prices = n.statistics.revenue(
        comps=["Load"],
        groupby=grouper,
        aggregate_time=False,
    ) / (
        -1
        * n.statistics.withdrawal(
            comps=["Load"],
            groupby=grouper,
            aggregate_time=False,
        )
    )
    hourly_prices = hourly_prices.reset_index()

    return hourly_prices


# %%
if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "export_to_REMIND",
            configfiles="resources/PyPSA_PkBudg1000_DEU_rm350_pypsa202504_bounds_2025-05-07_11.04.32/i1/config.remind_scenario.yaml",
            iteration="1",
            scenario="PyPSA_PkBudg1000_DEU_rm350_pypsa202504_bounds_2025-05-07_11.04.32",
        )

        # Manual input for testing
        fp_networks = [
            f"../results/{snakemake.wildcards['scenario']}/i{snakemake.wildcards['iteration']}/y2050/networks/base_s_4_elec_3H-Ep137.1.nc",
        ]
        fp_triggers_op = [
            #f"../results/{snakemake.wildcards['scenario']}/i{snakemake.wildcards['iteration']}/y2030/networks/elec_s_4_ec_lcopt_3H-Ep131.8_op_trigger",
        ]
        fp_triggers_op_perturb = [
            #f"../results/{snakemake.wildcards['scenario']}/i{snakemake.wildcards['iteration']}/y2030/networks/elec_s_4_ec_lcopt_3H-Ep131.8_op_perturb_biomass_trigger",
            f"../results/{snakemake.wildcards['scenario']}/i{snakemake.wildcards['iteration']}/y2030/networks/elec_s_4_ec_lcopt_3H-Ep131.8_op_perturb_CCGT_trigger",
        ]
    else:
        fp_networks = snakemake.input["networks"]  # For testing this doesn't work
        fp_triggers_op = snakemake.input["triggers_op"]
        fp_triggers_op_perturb = snakemake.input["triggers_op_perturb"]

    configure_logging(snakemake)

    # Get PyPSA-EUR to general technology mapping
    map_pypsaeur_to_general = get_pypsa_to_general_mapping(
        snakemake.input["technology_cost_mapping"]
    )

    # Get general to REMIND technology mapping
    map_general_to_remind = get_general_to_remind_mapping(
        snakemake.input["technology_cost_mapping"]
    )

    # Manually define mapping for loads
    # TODO: Define elsewhere or remove
    map_pypsaeur_to_remind_loads = {
        "AC": ["AC"],
        "H2 demand REMIND": ["H2 demand REMIND"],
    }

    # Create region mapping
    region_mapping = get_pypsa_to_remind_region_mapping(
        snakemake.input["region_mapping"]
    )

    # Define dictionary of coupling parameters
    # The func key is the function to calculate the parameter
    # The params key provide parameters for the function
    # The gdx key is used to save the parameter to a GDX file
    coupling_functions = {
        "capacity_factors": {
            "func": calculate_capacity_factors,
            "params": {
                "comps": ["Generator", "Link"],
                "grouper": ["region", "general_carrier"],
                "map_to_remind": True,
            },
            "gdx": {
                "description": "Capacity factors of generators and links [1]",
                "dims": ["year", "region", "carrier"],
            },
        },
        "markups_supply": {
            "func": calculate_markups_supply,
            "params": {
                "comps": ["Generator"],
                "grouper": ["region", "general_carrier"],
                "z_cutoff": snakemake.config["remind_coupling"]["export_to_REMIND"][
                    "z_cutoff"
                ],
                "map_to_remind": True,
            },
            "gdx": {
                "description": "Markups of supply-side generators [$/MWh]",
                "dims": ["year", "region", "carrier"],
            },
        },
        "markups_demand": {
            "func": calculate_markups_demand,
            "params": {
                "grouper": ["region", "general_carrier"],
                "z_cutoff": snakemake.config["remind_coupling"]["export_to_REMIND"][
                    "z_cutoff"
                ],
                "map_to_remind": False,
            },
            "gdx": {
                "description": "Markups of demand-side end-users [$/MWh]",
                "dims": ["year", "region", "enduse"],
            },
        },
        "peak_residual_loads": {
            "func": calculate_peak_residual_loads,
            "params": {"grouper": "region", "kind": "relative"},
            "gdx": {
                "description": "Peak residual load relative to load [1]",
                "dims": ["year", "region"],
            },
        },
        "availability_factors": {
            "func": calculate_availability_factors,
            "params": {
                "comps": ["Generator"],
                "grouper": ["region", "general_carrier"],
                "map_to_remind": True,
            },
            "gdx": {
                "description": "Availability factors of generators [1]",
                "dims": ["year", "region", "carrier"],
            },
        },
        "generation_shares": {
            "func": calculate_generation_shares,
            "params": {
                "grouper": ["region", "general_carrier"],
                "year": "placeholder",
            },  # Year inserted in loop
            "gdx": {
                "description": "Generation shares of technologies [1]",
                "dims": ["year", "region", "carrier"],
            },
        },
        "potentials": {
            "func": calculate_potentials,
            "params": {"grouper": ["region", "general_carrier"], "map_to_remind": True},
            "gdx": {
                "description": "Potentials of renewable technologies [MW]",
                "dims": ["year", "region", "carrier"],
            },
        },
        # Capacities for links are w.r.t. input at bus0, not output at bus1
        # Need to multiply by efficiency in REMIND to get output capacity
        "optimal_capacities": {
            "func": calculate_optimal_capacities,
            "params": {
                "comps": ["Generator", "Link", "Store"],
                "grouper": ["region", "general_carrier"],
                "weigh_by_remind": True,
                "year": "placeholder",
            },  # Year inserted in loop
            "gdx": {
                "description": "Optimal capacities of technologies, ATTENTION for links w.r.t. input [MW or MWh]",
                "dims": ["year", "region", "carrier"],
            },
        },
        "hydrogen_storage_generation": {
            "func": calculate_link_generation,
            "params": {
                "carrier": "H2 fuel cell",  # TODO: Use mapping
                "grouper": "region",
                "kind": "relative",
            },
            "gdx": {
                "description": " Hydrogen turbine generation relative to load [1]",
                "dims": ["year", "region"],
            },
        },
        "battery_storage_generation": {
            "func": calculate_link_generation,
            "params": {
                "carrier": "battery discharger",  # TODO: Use mapping
                "grouper": "region",
                "kind": "relative",
            },
            "gdx": {
                "description": "Battery generation relative to load [1]",
                "dims": ["year", "region"],
            },
        },
        "grid_losses": {
            "func": calculate_grid_losses,
            "params": {"grouper": "region", "kind": "relative"},
            "gdx": {
                "description": "Grid losses relative to load [1]",
                "dims": ["year", "region"],
            },
        },
        "difference_quotient_capacity_factors": {
            "func": calculate_difference_quotient,
            "params": {
                "property_func": calculate_capacity_factors,
                "comps": ["Generator"],
                "grouper": ["region", "general_carrier"],
                "exclude": ["nuclear", "hydro"],  # TODO: Put in config
                "map_to_remind": False,  # Applies to property_func
            },
            "gdx": {
                "description": "Difference quotients of capacity factors w.r.t. capacity [1/MW]",
                "dims": ["year", "region", "carrier", "carrier"],
            },
        },
        "difference_quotient_markups_supply": {
            "func": calculate_difference_quotient,
            "params": {
                "property_func": calculate_markups_supply,
                "comps": ["Generator"],
                "grouper": ["region", "general_carrier"],
                "exclude": ["nuclear", "hydro"],  # TODO: Put in config
                "z_cutoff": snakemake.config["remind_coupling"]["export_to_REMIND"][
                    "z_cutoff"
                ],
                "map_to_remind": False,  # Applies to propert_func
            },
            "gdx": {
                "description": "Difference quotients of supply-side markups w.r.t. capacity [($/MWh)/MW]",
                "dims": ["year", "region", "carrier", "carrier"],
            },
        },
    }

    # Define dictionary of reporting parameters
    # The func key is the function to calculate the parameter
    # The params key provide parameters for the function
    reporting_functions = {
        # TODO: Add bus_carrier to all components to enable grouping for energy balances
        "energy_balance_ac": {
            "func": calculate_energy_balance,
            "params": {
                "grouper": ["region", "general_carrier"],
                "bus_carrier": "AC",
            },
        },
        "energy_balance_addH2": {
            "func": calculate_energy_balance,
            "params": {
                "grouper": ["region", "general_carrier"],
                "bus_carrier": "H2",
            },
        },
        "preinstalled_capacities": {
            "func": calculate_preinstalled_capacities,
            "params": {"grouper": ["region", "general_carrier"]},
        },
        "optimal_capacities": {
            "func": calculate_optimal_capacities,
            "params": {
                "comps": ["Generator", "Link", "Store", "StorageUnit", "Line"],
                "grouper": ["region", "general_carrier"],
                "weigh_by_remind": False,
            },
        },
        "peak_residual_load_absolute": {
            "func": calculate_peak_residual_loads,
            "params": {"grouper": "region", "kind": "absolute"},
        },
        "hydrogen_storage_generation_absolute": {
            "func": calculate_link_generation,
            "params": {
                "carrier": "H2 fuel cell",  # TODO: Use mapping
                "grouper": "region",
                "kind": "absolute",
            },
        },
        "battery_storage_generation_absolute": {
            "func": calculate_link_generation,
            "params": {
                "carrier": "battery discharger",  # TODO: Use mapping
                "grouper": "region",
                "kind": "absolute",
            },
        },
        "grid_losses_absolute": {
            "func": calculate_grid_losses,
            "params": {"grouper": "region", "kind": "absolute"},
        },
        "load_prices": {
            "func": calculate_load_prices_with_cutoff,
            "params": {
                "grouper": ["region", "general_carrier"],
                "z_cutoff": False,
            },
        },
        # TODO: Make cutoff another column in load_prices instead
        "load_prices_cutoff": {
            "func": calculate_load_prices_with_cutoff,
            "params": {
                "grouper": ["region", "general_carrier"],
                "z_cutoff": snakemake.config["remind_coupling"]["export_to_REMIND"][
                    "z_cutoff"
                ],
            },
        },
        "market_values_supply": {
            "func": calculate_market_values_supply,
            "params": {
                "comps": ["Generator"],
                "grouper": ["region", "general_carrier"],
                "z_cutoff": False,
            },
        },
        # TODO: Make cutoff another column in market_values_supply instead
        "market_values_supply_cutoff": {
            "func": calculate_market_values_supply,
            "params": {
                "comps": ["Generator"],
                "grouper": ["region", "general_carrier"],
                "z_cutoff": snakemake.config["remind_coupling"]["export_to_REMIND"][
                    "z_cutoff"
                ],
            },
        },
        "market_values_demand": {
            "func": calculate_market_values_demand,
            "params": {"z_cutoff": False},
        },
        # TODO: Make cutoff another column in market_values_demand instead
        "market_values_demand_cutoff": {
            "func": calculate_market_values_demand,
            "params": {
                "z_cutoff": snakemake.config["remind_coupling"]["export_to_REMIND"][
                    "z_cutoff"
                ]
            },
        },
        "curtailments": {
            "func": calculate_curtailments,
            "params": {"grouper": ["region", "general_carrier"]},
        },
        "hourly_prices": {
            "func": calculate_hourly_prices,
            "params": {"grouper": ["region", "general_carrier"]},
        },
    }

    # Initialise empty dictionaries to store coupling and reporting parameters
    coupling_parameters = {}
    reporting_parameters = {}

    remind_coupling = snakemake.params.get("remind_settings")

    # Operation networks
    if fp_triggers_op:
        logger.info("Checking which operational networks are available.")
        # Get the list of operational network file paths by replacing "_trigger" with ".nc"
        fp_networks_op = [fp.replace("_trigger", ".nc") for fp in fp_triggers_op]
        # Sleep for 5 seconds to ensure all files are available
        time.sleep(5)
        # Check which operational network files exist
        fp_networks_op_available = [fp for fp in fp_networks_op if os.path.exists(fp)]
        # Log warnings for any missing files
        missing_networks_op = set(fp_networks_op) - set(fp_networks_op_available)
        for missing in missing_networks_op:
            logger.warning(f"Operational network {missing} not found. Skipping.")
    else:
        fp_networks_op_available = []

    # Perturbed networks
    if fp_triggers_op_perturb:
        logger.info("Checking which perturbed networks are available.")
        # Get the list of perturbed network file paths by replacing "_trigger" with ".nc"
        fp_networks_perturb = [
            fp.replace("_trigger", ".nc") for fp in fp_triggers_op_perturb
        ]
        # Sleep for 5 seconds to ensure all files are available
        time.sleep(5)
        # Check which perturbed network files exist
        fp_networks_perturb_available = [
            fp for fp in fp_networks_perturb if os.path.exists(fp)
        ]
        # Log warnings for any missing files
        missing_networks_perturb = set(fp_networks_perturb) - set(
            fp_networks_perturb_available
        )
        for missing in missing_networks_perturb:
            logger.warning(f"Perturbed network {missing} not found. Skipping.")
    else:
        fp_networks_perturb_available = []

    # Combine all networks
    fp_networks_all = (
        fp_networks + fp_networks_op_available + fp_networks_perturb_available
    )

    # Create dataframe containing metadata of all networks in this iteration
    networks = pd.DataFrame(fp_networks_all, columns=["filepath"])
    networks["year"] = networks["filepath"].str.extract(r"y(\d{4})")
    networks["op"] = networks["filepath"].str.contains("_op")
    networks["perturbed"] = networks["filepath"].str.contains("op_perturb_")
    networks["ptech"] = networks["filepath"].str.extract(r"op_perturb_(\w+).nc")
    networks["ref"] = ~networks["perturbed"]
    networks = networks.sort_values("year")

    networks_ref = networks.query("ref")
    networks_ptech = networks.query("perturbed")

    if remind_coupling["export_to_REMIND"]["use_operations_network"]:
        if remind_coupling["solve_operations_network"]["enable"]:
            # Group by year and check whether within that group the op=True network is available. If it is, drop the op=False network
            networks_ref = (
                networks_ref.groupby("year")
                .apply(lambda x: x.drop(x[(x["op"] == False)].index))
                .reset_index(drop=True)
            )
        else:
            logger.warning(
                "export_to_REMIND.use_operations_network also requires solve_operations_network.enable."
            )
    else:
        # Drop operational networks
        networks_ref = networks_ref[~networks["op"]].reset_index(drop=True)

    networks = pd.concat([networks_ref, networks_ptech], ignore_index=True)

    # Check if there is exactly one reference network per year
    net_ref_sum = networks.groupby("year")["ref"].sum()
    if not net_ref_sum.eq(1).all():
        raise ValueError(
            f"Expected exactly one reference network per year, but found {net_ref_sum}"
        )

    # Print unique years
    logger.info(f"Unique years in networks: {networks['year'].unique()}")

    # Loop over all years
    for year, df in networks.groupby("year"):
        logger.info(f"Processing year {year}, file path: {df['filepath'].values[0]}")
        # Load reference network and add region and general_carrier to network components
        n = pypsa.Network(df.query("ref")["filepath"].values[0])
        # If the network has no objective, raise a warning and skip
        if not hasattr(n, "objective"):
            logger.warning(
                f"Network {df.query('ref')['filepath'].values[0]} in year {year} has no objective. The solving probably failed. Skipping."
            )
            continue
        add_columns_for_processing(n, region_mapping, map_pypsaeur_to_general)
        check_for_mapping_completeness(n)

        # Calculate and store coupling parameters
        for key, values in coupling_functions.items():
            # Only calculate non-difference quotients parameters
            if values["func"] != calculate_difference_quotient:
                func, params = values["func"], values["params"]
                # Call function, injecting 'year' if needed
                result = func(
                    n, **{**params, "year": year} if "year" in params else params
                )
                # Insert year in first column
                result.insert(0, "year", year)
                # Concatenate data with previous years
                if key in coupling_parameters:
                    coupling_parameters[key] = pd.concat(
                        [coupling_parameters[key], result], ignore_index=True
                    )
                else:
                    coupling_parameters[key] = result

        # Calculate and store reporting parameters
        for key, values in reporting_functions.items():
            func, params = values["func"], values["params"]
            result = func(n, **params)
            # Insert year in first column
            result.insert(0, "year", year)
            # Concatenate data with previous years
            if key in reporting_parameters:
                reporting_parameters[key] = pd.concat(
                    [reporting_parameters[key], result], ignore_index=True
                )
            else:
                reporting_parameters[key] = result

        # For each year, calculate difference quotients for perturbed networks (if available)
        for p in df.query("perturbed")["ptech"]:

            # Load perturbed network and add region and general_carrier to network components
            npert = pypsa.Network(df.query(f"ptech == '{p}'")["filepath"].values[0])
            add_columns_for_processing(npert, region_mapping, map_pypsaeur_to_general)
            check_for_mapping_completeness(npert)

            # Calculate difference quotients
            for key, values in coupling_functions.items():
                # Now only calculate difference quotients
                if values["func"] == calculate_difference_quotient:
                    func, params = values["func"], values["params"]
                    result = func(n_opt=n, n_pert=npert, ptech=p, **params)
                    # Insert year in first column
                    result.insert(0, "year", year)
                    # Concatenate data
                    if key in coupling_parameters:
                        coupling_parameters[key] = pd.concat(
                            [coupling_parameters[key], result], ignore_index=True
                        )
                    else:
                        coupling_parameters[key] = result

    # Write coupling parameters to GDX
    logger.info("Writing coupling parameters to GDX file")

    # Create GDX container
    gdx = gt.Container()

    # Define GDX sets
    sets_definition = {
        "year": {
            "records": networks["year"].unique(),
            "description": "Years in which PyPSA networks were solved",
        },
        "region": {
            "records": region_mapping.loc[snakemake.config["countries"]].iloc[0],
            "description": "REMIND regions for which PyPSA networks were solved",
        },
        "carrier": {
            "records": list(
                set(
                    [
                        item
                        for sublist in map_general_to_remind.values()
                        for item in sublist
                    ]
                )
            ),
            "description": "REMIND technologies for which PyPSA networks were solved",
        },
        "enduse": {
            "records": coupling_parameters["markups_demand"][
                "general_carrier"
            ].unique(),
            "description": "REMIND end-use sectors for which PyPSA loads were disaggregated",
        },
    }

    # Create sets
    sets = {}
    for name, params in sets_definition.items():
        sets[name] = gt.Set(
            gdx, name, records=params["records"], description=params["description"]
        )

    # Add all coupling parameters to GDX
    for key, df in coupling_parameters.items():
        gt.Parameter(
            gdx,
            name=key,
            domain=coupling_functions[key]["gdx"]["dims"],
            records=df,
            description=coupling_functions[key]["gdx"]["description"],
        )

    # Write GDX file
    gdx.write(snakemake.output["gdx"])

    # Export coupling parameters to CSV
    # snakemake.output["coupling_parameters"] gives the direcory
    os.makedirs(snakemake.output["coupling_parameters"], exist_ok=True)

    for key, df in coupling_parameters.items():
        df.to_csv(snakemake.output["coupling_parameters"] + f"/{key}.csv", index=False)

    # Export reporting parameters to CSV
    # snakemake.output["reporting_parameters"] gives the direcory
    os.makedirs(snakemake.output["reporting_parameters"], exist_ok=True)

    for key, df in reporting_parameters.items():
        df.to_csv(snakemake.output["reporting_parameters"] + f"/{key}.csv", index=False)

    # Sleep for 5 seconds to ensure all files are written
    time.sleep(5)

# %%
