# -*- coding: utf-8 -*-
# %%
import logging
import re

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

def add_region_and_general_carrier(network, region_mapping, map_pypsaeur_to_general):
    """
    Add column region and general_carrier to network components.
    """
    # Remove columns from network components if they already exist
    # These come from the RCL constraints
    # TODO: Harmonise columns names in RCL constraints
    for comp in ["generators", "links", "stores"]:
        if "region_REMIND" in getattr(network, comp).columns:
            getattr(network, comp).drop(columns=["region_REMIND"], inplace=True)
        if "technology_group" in getattr(network, comp).columns:
            getattr(network, comp).drop(columns=["technology_group"], inplace=True)
    
    # Add region to buses if it doesnt exist (this is the case if additionakl h2demand is not enabled)
    if "region" not in network.buses.columns:
        network.buses["region"] = ""

    # First map the PyPSA-EUR countries to REMIND-EU regions;
    # .statistics(..) can then automatically take care of the aggregation
    # H2 demand buses already have a region assigned, so we don't want to overwrite those
    network.buses["region"] = network.buses["region"].where(
        network.buses["region"] != "",
        network.buses["country"].map(region_mapping["REMIND-EU"]),
    )

    # Add information for aggregation later: region name (REMIND-EU) and general carrier
    network.generators["region"] = network.generators["bus"].map(
        network.buses["region"]
    )
    network.stores["region"] = network.stores["bus"].map(network.buses["region"])
    network.storage_units["region"] = network.storage_units["bus"].map(
        network.buses["region"]
    )
    network.links["region"] = network.links["bus0"].map(network.buses["region"])
    network.lines["region"] = network.lines["bus0"].map(network.buses["region"])
    network.loads["region"] = network.loads["bus"].map(network.buses["region"])
    # Links/lines have two buses, and can be attributed to two regions (used for e.g. grid length calculations)
    network.links["region1"] = network.links["bus1"].map(network.buses["region"])
    network.lines["region1"] = network.lines["bus1"].map(network.buses["region"])
    
    # Add general carrier to network components
    network.generators["general_carrier"] = network.generators["carrier"].map(
        map_pypsaeur_to_general
    )
    network.stores["general_carrier"] = network.stores["carrier"]
    network.storage_units["general_carrier"] = network.storage_units["carrier"].map(
        map_pypsaeur_to_general
    )
    network.links["general_carrier"] = network.links["carrier"]
    network.lines["general_carrier"] = network.lines["carrier"]
    network.loads["general_carrier"] = network.loads["bus"].map(
        network.buses["carrier"]
    )
    
def get_pypsa_to_general_mapping(fp_mapping):
    """
    Get mapping from PyPSA-EUR to REMIND-EU technologies.
    """
    map_pypsaeur_to_general = (
        get_technology_mapping(
            fp_mapping, group_technologies=True
        )
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
        get_technology_mapping(
            fp_mapping, group_technologies=True
        )
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

    if (
        tmp_set := set(n.loads["general_carrier"])
        - map_pypsaeur_to_remind_loads.keys()
    ):
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
    removes excess columns / sets index and sorts by region + year.

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
        df = df.query("general_carrier != 'H2 transfer to H2 demand REMIND'")
        df = df.query("general_carrier != 'H2 demand buffer REMIND'")
    
    # Helper function to map carriers to REMIND technologies
    def map_carriers_for_remind(dg, col):
        """
        Maps the carrier names from general technologies to REMIND
        technologies.
        """
        # Mapping
        old_carrier = dg.iloc[0][col]
        new_carriers = map_general_to_remind[old_carrier]
        # Repeat rows for each new carrier, create new dataframe then assign the new carrier name
        dg = pd.DataFrame(
            np.repeat(dg.values, len(new_carriers), axis=0), columns=dg.columns
        )
        dg[col] = new_carriers
        return dg
    
    # Map carriers to REMIND technologies for general_carrier
    if map_to_remind and "carrier_perturbed" not in df.columns:
        df = df.rename(columns={"general_carrier": "carrier"})
        df = df.groupby(["region", "carrier"], group_keys=False).apply(
            map_carriers_for_remind, col="carrier"
        )
    # Map carriers to REMIND technologies for both carrier and carrier_perturbed
    if map_to_remind and "carrier_perturbed" in df.columns:
        df = df.rename(columns={"general_carrier": "carrier"})
        df = df.groupby(["region", "carrier", "carrier_perturbed"], group_keys=False).apply(
            map_carriers_for_remind, col="carrier"
        )
        df = df.groupby(["region", "carrier", "carrier_perturbed"], group_keys=False).apply(
            map_carriers_for_remind, col="carrier_perturbed"
        )

    return df

def postprocess_dataframe(df, map_to_remind=True):
    """
    General function to postprocess the dataframes, combines the network-
    specific results into one dataframe removes excess columns / sets index
    and sorts by region + year.

    map_to_remind: bool, default True
        Whether to map the general carrier names to REMIND carrier names.
    """
    df = pd.concat(df)
    df = df.rename(
        columns={"general_carrier": "carrier"}
    )  # different auxiliary columns have different names; rename for consistency
    
    # Set index dynamically
    if "carrier" in df.columns:
        df = df.set_index(
            ["year", "region", "carrier"]
        ).sort_index()        
    elif "perturbed_carrier" in df.columns:
        df = df.set_index(
            ["year", "region", "carrier", "perturbed_carrier"]
        ).sort_index()
    else:
        df = df.set_index(
            ["year", "region"]
        ).sort_index()

    # Reset index
    df = df.reset_index().drop(
        columns=["level_0"],
        errors="ignore",
    )

    if "carrier" in df.columns:
        # Remove rows related to the additional hydrogen bus
        df = df.query("carrier != 'H2 transfer to H2 demand REMIND' and carrier != 'H2 demand buffer REMIND'")
        # Remove 
    
    def map_carriers_for_remind(dg, col="carrier"):
        """
        Maps the carrier names from general technologies to REMIND
        technologies.
        """
        old_carrier = dg.iloc[0][col]

        # Mapping for generators and loads are different
        if old_carrier in map_general_to_remind.keys():
            _map = map_general_to_remind

        new_carriers = _map[old_carrier]

        # Repeat rows for each new carrier, create new dataframe then assign the new carrier name
        dg = pd.DataFrame(
            np.repeat(dg.values, len(new_carriers), axis=0), columns=dg.columns
        )
        dg[col] = new_carriers
        return dg

    if map_to_remind and "carrier_perturbed" not in df.columns:
        # Map carriers to REMIND technologies
        df = df.groupby(["year", "region", "carrier"], group_keys=False).apply(
            map_carriers_for_remind, col = "carrier"
            )
    if map_to_remind and "carrier_perturbed" in df.columns:
        # Map carriers to REMIND technologies
        df = df.groupby(["year", "region", "carrier", "carrier_perturbed"], group_keys=False).apply(
            map_carriers_for_remind, col = "carrier"
            )
        df = df.groupby(["year", "region", "carrier", "carrier_perturbed"], group_keys=False).apply(
            map_carriers_for_remind, col = "carrier_perturbed"
            )

    return df

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
    capacity_factors = n.statistics.capacity_factor(
        comps=comps, groupby=grouper
    )
    capacity_factors = capacity_factors.to_frame("value").reset_index().drop(columns=["component"])
       
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
    load_prices = (
        n.statistics.revenue(
            comps=["Load"], groupby=grouper
        )
        / (
            -1
            * n.statistics.withdrawal(
                comps=["Load"], groupby=grouper
            )
        )
    )
    load_prices = load_prices.to_frame("value").reset_index().drop(columns=["component"])
    
    return load_prices

def calculate_markups_supply(n, comps, grouper, map_to_remind):
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
    map_to_remind: bool
        Whether to map the general carrier names to REMIND carrier names.
    """
    # Calculate markups for the supply side
    market_value = n.statistics.market_value(
        comps=comps, groupby=grouper
    )
    
    # Get average electricity price
    load_price_ac = calculate_load_prices(n, grouper).query("general_carrier == 'AC'").value[0]
    
    # Subtract average electricity price from market value to get markup
    markups_supply = market_value - load_price_ac
    markups_supply = markups_supply.to_frame("value").reset_index().drop(columns=["component"])
    
    return process_data(markups_supply, cols=grouper, map_to_remind=map_to_remind)

# TODO: Make compatible with multiple regions
def calculate_markups_demand(n, grouper, map_to_remind):
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
    """
    # Electrolysis generation series
    gen = n.links_t.p0.loc[:, n.links.carrier == "H2 electrolysis"]
    gen.columns = gen.columns.map(n.links.bus0)
    gen = gen.T.groupby(level=0).sum().T
    # Local marginal price
    lmp = n.buses_t.marginal_price.loc[:,gen.columns]
    # Calculate market value
    mv = (gen * lmp).sum().sum() / gen.sum().sum()
    
    # Get average electricity price
    load_price_ac = calculate_load_prices(n, grouper).query("general_carrier == 'AC'").value[0]
    
    # Subtract average electricity price from market value to get markup
    markup_electrolysis = mv - load_price_ac
    
    # Create DataFrame
    markups_demand = pd.DataFrame(
        {
            "region": ["DEU"],
            "general_carrier": ["electrolysis"],
            "value": [markup_electrolysis],
        }
    )
    
    return process_data(markups_demand, cols=grouper, map_to_remind=map_to_remind)

# TODO: Implement output of both absolute and relative peak residual loads
def calculate_peak_residual_loads(n, grouper, kind, map_to_remind):
    """
    Calculate peak residual loads for the network.

    Parameters
    ----------
    n : pypsa.Network
        Network to calculate peak residual loads for.
    kind: str
        Kind of peak residual load to calculate. Can be "absolute" or "relative".
    """
    ## Calculate peak residual load
    dispatchable_technologies = set(n.generators.index) - set(n.generators_t.p_max_pu.columns)
    # Add attribute to network.generators to distinguish between dispatchable and non-dispatchable technologies
    n.generators["peak_residual_load"] = "No"
    n.generators.loc[
        list(dispatchable_technologies), "peak_residual_load"
    ] = "Yes"
    # Don't include load shedding as dispatchable technology
    n.generators.loc[n.generators.index.str.contains("load"), "peak_residual_load"] = "No"
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
                "absolute": x.xs("Yes", level="peak_residual_load")[
                    max_prl_snapshot
                ]
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
    # Rename to value
    peak_residual_load.columns = [grouper, "value"]
    
    return process_data(peak_residual_load, cols=grouper, map_to_remind=map_to_remind)

# Non-standard function, following standard implementation from PyPSA.statistics
def calculate_availability_factors(
    n,
    comps=None,
    grouper=None,
):
    """
    Calculate the availability factor of components in the network.

    For information on the list of arguments, see the docs in
    `Network.statistics` or `pypsa.statistics.StatisticsAccessor`.

    Parameters
    ----------
    aggregate_time : str, bool, optional
        Type of aggregation when aggregating time series.
        Note that for {'mean', 'sum'} the time series are aggregated to
        using snapshot weightings. With False the time series is given. Defaults to 'mean'.
    """

    def get_availability(n, c):
        """
        Get the availability time series of a component.

        For generators with p_max_pu time-series (usually renewable
        generators) this is p_max_pu * p_nom_opt, for conventional
        generators it is just the dispatch p time-series like for the
        capacity factor.
        """
        if c in n.branch_components:
            return n.pnl(c).p0
        elif c == "Store":
            return n.pnl(c).e
        else:
            p = n.pnl(c).p.copy(deep=True)
            p_max_pu = n.pnl(c).p_max_pu * n.generators.p_nom_opt
            p.update(p_max_pu)
            return p

    def func(n, c, port):
        p = get_availability(n, c).abs()
        weights = pypsa.statistics.get_weightings(n, c)
        return pypsa.statistics.aggregate_timeseries(p, weights, agg="mean")

    # HACK to use the _aggregate_components function
    statistics_accessor = pypsa.statistics.StatisticsAccessor(n)
    statistics_accessor.n = n
    
    df = statistics_accessor._aggregate_components(
        func, comps=comps, agg="sum", groupby=grouper
    )

    capacity = n.statistics.optimal_capacity(
        comps=comps, aggregate_groups="sum", groupby=grouper
    )
    df = df.div(capacity, axis=0)
    df = df.to_frame("value").reset_index()
    return df

def calculate_potentials(n, grouper):
    """
    Calculate VRE potentials for the network.

    Parameters
    ----------
    n : pypsa.Network
        Network to calculate potentials for.
    grouper : list
        List of columns to group the potentials by.
    """
    # RCL generators have to be excluded from potentials
    df = n.generators.copy(deep=True)
    df = df.query("not index.str.contains('RCL')", engine="python")
    potential = df.groupby(grouper)["p_nom_max"].sum()
    potential = potential.replace([np.inf, -np.inf], np.nan).dropna()
    potential = potential.to_frame("value").reset_index()
    
    return potential

def calculate_optimal_capacities(n, comps, grouper):
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
    """
    # Calculate optimal capacities
    optimal_capacities = n.statistics.optimal_capacity(
        comps=comps, groupby=grouper
    )
    optimal_capacities = (
        optimal_capacities.to_frame("value").reset_index()
    )
    return optimal_capacities

def calculate_grid_losses(n, kind="relative"):
    """
    Calculate grid losses for the network.

    Parameters
    ----------
    n : pypsa.Network
        Network to calculate grid losses for.
    """
    ## Determine grid losses in absolute and relative terms
    grid_loss_abs = n.statistics.energy_balance(comps = "Line", groupby="region").abs()
    # Handle the case where there are no grid losses
    if grid_loss_abs.empty:
        regions = n.buses["region"].unique()
        grid_loss_abs = pd.Series(0, index=regions, name="absolute")
        grid_loss_abs.index.name = "region"
    grid_loss_rel = grid_loss_abs / n.statistics.withdrawal(comps="Load", bus_carrier="AC", groupby="region")
    
    grid_loss = pd.DataFrame({
        "absolute": grid_loss_abs,
        "relative": grid_loss_rel
    }).reset_index()
    
    grid_loss = grid_loss[["region", kind]]
    
    return grid_loss

# Helper function
def get_supply_with_zeros(n, carrier, component="Link"):
    """
    Get the supply data for a specific carrier, ensuring missing values are filled with zeros.

    Parameters
    ----------
    n : pypsa.Network
        The network object.
    carrier : str
        The carrier to extract (e.g., "H2 fuel cell").
    component : str, optional
        The component to filter by (default is "Link").

    Returns
    -------
    pd.Series
        Supply data for the specified carrier with zeros filled.
    """
    # Extract supply data for the specified component
    supply_data = n.statistics.supply(comps=[component], groupby=["region", "carrier"])
    supply_data = supply_data.xs(component, level="component")  # Filter by component

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
    supply_series = supply_series.reorder_levels(["component", "region"])["objective"]

    return supply_series

def calculate_hydrogen_storage_generation(n, kind="relative"):
    """
    Calculate hydrogen storage generation for the network.

    Parameters
    ----------
    n : pypsa.Network
        Network to calculate hydrogen storage generation for.
    """
    # Get absolute supply for "H2 fuel cell"
    absolute_supply = get_supply_with_zeros(n, carrier="H2 fuel cell")

    # Calculate relative supply
    relative_supply = absolute_supply / n.statistics.withdrawal(
        comps="Load", bus_carrier="AC", groupby="region"
    )

    # Combine absolute and relative supply into a DataFrame
    h2_turbine_storage = pd.DataFrame({
        "absolute": absolute_supply,
        "relative": relative_supply.fillna(0)  # Fill NaN values with 0
    }).reset_index().drop(columns=["component"])
    
    h2_turbine_storage = h2_turbine_storage[["region", kind]]

    return h2_turbine_storage

def calculate_battery_storage_generation(n, kind="relative"):
    """
    Calculate battery storage generation for the network.

    Parameters
    ----------
    n : pypsa.Network
        Network to calculate battery storage generation for.
    """
    # Get absolute supply for "battery discharger"
    absolute_supply = get_supply_with_zeros(n, carrier="battery discharger")

    # Calculate relative supply
    relative_supply = absolute_supply / n.statistics.withdrawal(
        comps="Load", bus_carrier="AC", groupby="region"
    )

    # Combine absolute and relative supply into a DataFrame
    battery_storage = pd.DataFrame({
        "absolute": absolute_supply,
        "relative": relative_supply.fillna(0)  # Fill NaN values with 0
    }).reset_index().drop(columns=["component"])
    
    battery_storage = battery_storage[["region", kind]]

    return battery_storage

def calculate_generation_shares(n, grouper):
    """
    Calculate generation shares for the network.

    Parameters
    ----------
    n : pypsa.Network
        Network to calculate generation shares for.
    grouper : list
        List of columns to group the generation shares by.
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
    generation_shares = generation_share.to_frame("value").reset_index().drop(columns=["component"])
    
    return generation_shares

def difference_quotient(n_opt, n_pert, ptech, property_func, grouper=None, **kwargs):
    """
    Compute the difference quotient as a numerical approximation of the partial
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
        Function for which the 
    grouper : list
        List of columns to group the generation shares by.
    """
    
    def get_filtered_capacity(n, ptech):
        # Calculate optimal capacity, convert to DataFrame, and filter by general_carrier
        return (n.statistics.optimal_capacity(comps="Generator", groupby=["region", "general_carrier"])
                .to_frame()
                .reset_index()
                .loc[lambda x: x["general_carrier"].str.contains(ptech, case=False, na=False)])

    # Get the property values for optimal and perturbed networks
    prop_opt = property_func(n_opt, grouper=grouper, **kwargs)
    prop_pert = property_func(n_pert, grouper=grouper, **kwargs)
    merged = prop_opt.merge(prop_pert, on=grouper, suffixes=("_orig", "_pert"))

    # Extract total capacities for the perturbed technology
    cap_orig = get_filtered_capacity(n_opt, ptech)
    cap_pert = get_filtered_capacity(n_pert, ptech)
    cap_merged = cap_orig.merge(cap_pert, on=grouper, suffixes=("_orig", "_pert"))
    cap_merged.rename(columns={"general_carrier": "carrier_perturbed"}, inplace=True)

    # Merge capacity with property values
    merged = merged.merge(cap_merged, on="region")

    # Compute anticipation factor (partial derivative approximation)
    merged["value"] = (
        (merged["value_pert"] - merged["value_orig"]) /
        (merged["p_nom_opt_pert"] - merged["p_nom_opt_orig"])
    )

    return merged[["region", "general_carrier", "carrier_perturbed", "value"]]


# Currently not used
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
    p0 = pd.concat([network.links_t["p0"], network.lines_t["p0"]], axis="columns")[relevant_connectors.index]
    p1 = pd.concat([network.links_t["p1"], network.lines_t["p1"]], axis="columns")[relevant_connectors.index]

    # Map relevant_connectors to buses, from which marginal prices are taken
    p0.columns = pd.MultiIndex.from_frame(
        relevant_connectors.loc[p0.columns, ["bus0", "bus1"]],
        names=["from", "to"],
    )
    p1.columns = pd.MultiIndex.from_frame(
        relevant_connectors.loc[p1.columns, ["bus1", "bus0"]],  # Reverse order as p1 is the reverse flow
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
    expense_import.index = (
        pd.MultiIndex.from_arrays([
            expense_import.index.get_level_values("from")
            .map(network.buses["region"]),
            expense_import.index.get_level_values("to")
            .map(network.buses["region"])
        ], names=["from", "to"])
    )

    revenue_export.index = (
        pd.MultiIndex.from_arrays([
            revenue_export.index.get_level_values("from")
            .map(network.buses["region"]),
            revenue_export.index.get_level_values("to")
            .map(network.buses["region"])
        ], names=["from", "to"])
    )

    # Sum over hours
    expense_import = expense_import.groupby(["from", "to"]).sum().sum(axis="columns")
    revenue_export = revenue_export.groupby(["from", "to"]).sum().sum(axis="columns")

    # Transpose
    p = p.T
    
    # Map buses to regions
    p.index = (
        pd.MultiIndex.from_arrays([
            p.index.get_level_values("from")
            .map(network.buses["region"]),
            p.index.get_level_values("to")
            .map(network.buses["region"])
        ], names=["from", "to"])
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

def calculate_market_values_supply(n, groupby):
    """
    Calculate market values for the all generators.

    Parameters
    ----------
    n : pypsa.Network
        Network to calculate market values for.
    groupby : list
        List of columns to group the market values by.
    """
    # Calculate market values after applying cutoff for electricity prices
    if cutoff_market_values := snakemake.config["remind_coupling"][
        "extract_coupling_parameters"
    ]["cutoff_market_values"]:
        relevant_buses = network.buses.query("carrier == 'AC'").index
        cutoff_market_values = float(cutoff_market_values)
        zscores = (
            network.buses_t["marginal_price"][relevant_buses]
            .apply(zscore)
            .mean(axis="columns")
        )
        zscores.index = pd.to_datetime(zscores.index)

        # By setting snapshot_weightings to 0, the market value will not be calculated for these snapshots above the cutoff value
        network.snapshot_weightings = network.snapshot_weightings.where(
            zscores < cutoff_market_values, 0
        )

        logger.info(
            "Cutoff for electricity prices in market value calculation enabled. "
            "Excluding {n} snapshots from calculations with electricity prices above {p:.2f} USD/MWh.".format(
                n=int(
                    network.snapshot_weightings["generators"].shape[0]
                    * network.snapshot_weightings["generators"].iloc[0]
                    - network.snapshot_weightings["generators"].sum()
                ),
                p=network.buses_t["marginal_price"][relevant_buses]
                .where(zscores < cutoff_market_values)
                .mean(axis="columns")
                .max(),
            )
        )

    # Calculate the market values (round-about way as the intended method of the statistics module is not yet available)
    market_values = network.statistics.market_value(
        comps=["Generator"],
        groupby=["region", "general_carrier"],
    )
    market_values = (
        market_values.to_frame("value").reset_index().drop(columns=["component"])
    )
    
    return market_values

#%%
if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "extract_coupling_parameters",
            configfiles="config/config.remind.yaml",
            iteration="9",
            scenario="TESTsmk",
            year="2030"
        )

        # Manual input for testing
        networks = [
            f"../results/{snakemake.wildcards['scenario']}/i{snakemake.wildcards['iteration']}/y{snakemake.wildcards['year']}/networks/elec_s_1_ec_lcopt_3H-Ep131.8.nc",
            f"../results/{snakemake.wildcards['scenario']}/i{snakemake.wildcards['iteration']}/y{snakemake.wildcards['year']}/networks/elec_s_1_ec_lcopt_3H-Ep131.8_op.nc",
            f"../results/{snakemake.wildcards['scenario']}/i{snakemake.wildcards['iteration']}/y{snakemake.wildcards['year']}/networks/elec_s_1_ec_lcopt_3H-Ep131.8_op_perturb_CCGT.nc",
            f"../results/{snakemake.wildcards['scenario']}/i{snakemake.wildcards['iteration']}/y{snakemake.wildcards['year']}/networks/elec_s_1_ec_lcopt_3H-Ep131.8_op_perturb_solar.nc",
        ]
    else:
        networks = snakemake.input["networks"]
        configure_logging(snakemake)

    # Get PyPSA-EUR to general technology mapping
    map_pypsaeur_to_general = get_pypsa_to_general_mapping(snakemake.input["technology_cost_mapping"])
    
    # Get general to REMIND technology mapping
    map_general_to_remind = get_general_to_remind_mapping(snakemake.input["technology_cost_mapping"])
    
    # Manually define mapping for loads
    # TODO: Define elsewhere?
    map_pypsaeur_to_remind_loads = {
        "AC": ["AC"],
        "H2 demand REMIND": ["H2 demand REMIND"],
    }
    
    # Create region mapping
    region_mapping = get_pypsa_to_remind_region_mapping(snakemake.input["region_mapping"])

    # Define coupling functions along with their required parameters
    coupling_functions_default = {
        "capacity_factors": {
            "func": calculate_capacity_factors,
            "params": {"comps": ["Generator", "Link"],
                       "grouper": ["region", "general_carrier"],
                       "map_to_remind": True},
        },
        "markups_supply": {
            "func": calculate_markups_supply,
            "params": {"comps": ["Generator"],
                       "grouper": ["region", "general_carrier"],
                       "map_to_remind": True},
        },
        "markups_demand": {
            "func": calculate_markups_demand,
            "params": {"grouper": ["region", "general_carrier"],
                       "map_to_remind": False},
        },
        "peak_residual_loads": {
            "func": calculate_peak_residual_loads,
            "params": {"grouper": "region",
                       "kind": "relative",
                       "map_to_remind": False},
        },
        "availability_factors": {
            "func": calculate_availability_factors,
            "params": {"comps": ["Generator"],
                       "grouper": ["region", "general_carrier"],
                       "map_to_remind": True},
        },
        "generation_shares": {
            "func": calculate_generation_shares,
            "params": {"grouper": ["region", "general_carrier"],
                       "map_to_remind": True},
        },
        "potentials": {
            "func": calculate_potentials,
            "params": {"grouper": ["region", "general_carrier"],
                       "map_to_remind": True},
        },
        "optimal_capacities": {
            "func": calculate_optimal_capacities,
            "params": {"comps": ["Generator", "Link", "Store"],
                       "grouper": ["region", "general_carrier"],
                       "map_to_remind": True},
        },
        "grid_losses": {
            "func": calculate_grid_losses,
            "params": {"kind": "relative",
                       "map_to_remind": False},
        },
        "hydrogen_storage_generation": {
            "func": calculate_hydrogen_storage_generation,
            "params": {"kind": "relative",
                       "map_to_remind": False},
        },
        "battery_storage_generation": {
            "func": calculate_battery_storage_generation,
            "params": {"kind": "relative",
                       "map_to_remind": False},
        }
    }
    
    # Define coupling functions for difference quotients
    coupling_functions_difference_quotient = {
        "difference_quotient_capacity_factors": {
            "func": difference_quotient,
            "params": {
                "property_func": calculate_capacity_factors,
                "comps": ["Generator"],
                "grouper": ["region", "general_carrier"],
                "map_to_remind": True
            },
        },
        "difference_quotient_markups_supply": {
            "func": difference_quotient,
            "params": {
                "property_func": calculate_markups_supply,
                "comps": ["Generator"],
                "grouper": ["region", "general_carrier"],
                "map_to_remind": True
            },
        }
    }

    # Initialise a dictionary for storing coupling parameters
    coupling_parameters = {}

    # Load perturbation settings
    perturbation = snakemake.params.get("perturbation")

    # Filter networks based on perturbation settings
    if perturbation["enable"] and perturbation["use_op_for_derivative"]:
        networks = [n for n in networks if "_op" in n]

    # Create dataframe containing metadata of all networks in this iteration
    networks = pd.DataFrame(networks, columns=["filepath"])
    networks["year"] = networks["filepath"].str.extract(r"y(\d{4})")
    networks["perturbed"] = networks["filepath"].str.contains("perturb")
    networks["ptech"] = networks["filepath"].str.extract(r"perturb_(\w+).nc")
    networks["ref"] = ~networks["perturbed"]

    # Ensure one reference network per year
    assert networks.groupby("year")["ref"].sum().eq(1).all(), "There must be exactly one reference network per year"

    # Loop over years
    for year, df in networks.groupby("year"):
        
        # Load reference network
        n = pypsa.Network(df.query("ref")["filepath"].values[0])
        
        # Add region and general_carrier to network components
        add_region_and_general_carrier(n, region_mapping, map_pypsaeur_to_general)
        check_for_mapping_completeness(n)

        # Calculate and store default coupling parameters
        for key, values in coupling_functions_default.items():
            func, params = values["func"], values["params"]
            result = func(n, **params)
            result.insert(0, "year", year)
            # Concatenate data
            if key in coupling_parameters:
                coupling_parameters[key] = pd.concat([coupling_parameters[key], result], ignore_index=True)
            else:
                coupling_parameters[key] = result
            
        # For each year, calculate difference quotients for perturbed networks (if available)
        for p in df.query("perturbed")["ptech"]:
            
            # Load perturbed network
            npert = pypsa.Network(df.query(f"ptech == '{p}'")["filepath"].values[0])
            
            # Add region and general_carrier to network components
            add_region_and_general_carrier(npert, region_mapping, map_pypsaeur_to_general)
            check_for_mapping_completeness(npert)
            
            # Calculate difference quotients
            for key, details in coupling_functions_difference_quotient.items():
                func, params = details["func"], details["params"]
                result = func(n_opt=n, n_pert=npert, ptech=p, **params)
                result["year"] = year
                # Concatenate data
                if key in coupling_parameters:
                    coupling_parameters[key] = pd.concat([coupling_parameters[key], result], ignore_index=True)
                else:
                    coupling_parameters[key] = result
    
                
#%%

























    # Initialise a dictionary for storing coupling parameters
    coupling_parameters = {
        "capacity_factors": [],
        "markups_supply": [],
        "markups_demand": [],
        "peak_residual_loads": [],
        "availability_factors": [],
        "generation_shares": [],
        "potentials": [],
        "optimal_capacities": [],
        "grid_losses": [],
#        "crossborder_flows": [],
#        "crossborder_prices_import": [],
#        "crossborder_prices_export": [],
#        "generation_region_shares": [],
        "hydrogen_storage_generation": [],
        "battery_storage_generation": [],
        "capacity_factors_derivative": [],
        "markups_supply_derivative": [],
        "peak_residual_loads_derivative": [],
    }

    # Values used only for reporting but not for coupling
    reporting_parameters = {
        "preinstalled_capacities": [],
        "energy_balances": [],
        "curtailments": [],
        "loads": [],
        "load_prices": [],
        "market_values_supply": [],
        "market_values_demand": [],
        "hourly_prices": [],
    }

    # Load perturbation settings
    perturbation = snakemake.params.get("perturbation")
    
    # Filter networks based on perturbation settings
    if perturbation["enable"] & perturbation["use_op_for_derivative"]:
        # Only include dispatch networks with "_op" in the filename
        networks = [n for n in networks if "_op" in n]

    # Create dataframe containing metadata of all networks in this iteration
    networks = pd.DataFrame(networks, columns=["filepath"])
    networks["year"] = networks["filepath"].str.extract(r"y(\d{4})")
    networks["perturbed"] = networks["filepath"].str.contains("perturb")
    networks["ptech"] = networks["filepath"].str.extract(r"perturb_(\w+).nc")
    networks["ref"] = ~networks["perturbed"]
    
    # Make sure there's only one reference network
    assert networks.groupby(["year"]).sum()["ref"].all() == 1, "There must be exactly one reference network per year"

#%%
    # Loop over years
    for year, df in networks.groupby("year"):
        
        # Load reference network
        n = pypsa.Network(df.query("ref")["filepath"].values[0])
        
        # Add region and general_carrier to network components
        add_region_and_general_carrier(n, region_mapping, map_pypsaeur_to_general)
        check_for_mapping_completeness(n)
        
        # Calculate capacity factors
        coupling_parameters["capacity_factors"] = [
            calculate_capacity_factors(
                n,
                comps=["Generator", "Link"],
                grouper=["region", "general_carrier"]
                )
        ]
        
        # Calculate market values (supply side)
        coupling_parameters["markups_supply"] = [
            calculate_markups_supply(
                n,
                comps=["Generator"],
                grouper=["region", "general_carrier"]
                )
        ]
        
        # Calculate market values (demand side)
        coupling_parameters["markups_demand"] = [
            calculate_markups_demand(
                n,
                grouper=["region", "general_carrier"]
                )
        ]
        
        # Caculate peak residual loads
        coupling_parameters["peak_residual_loads"] = [
            calculate_peak_residual_loads(
                n,
                grouper="region"
                )
        ]
        
        # Calculate availability factors
        coupling_parameters["availability_factors"] = [
            calculate_availability_factors(
                n,
                comps=["Generator"],
                grouper=["region", "general_carrier"]
                )
        ]
        
        # Calculate potentials
        coupling_parameters["potentials"] = [
            calculate_potentials(
                n,
                grouper=["region", "general_carrier"]
                )
        ]
        
        # Calculate optimal capacities
        coupling_parameters["optimal_capacities"] = [
            calculate_optimal_capacities(
                n,
                comps=["Generator", "Link", "Store"],
                grouper=["region", "general_carrier"]
                )
        ]
        
        # Calculate grid losses
        coupling_parameters["grid_losses"] = [
            calculate_grid_losses(n)
        ]
        
        # Calculate hydrogen storage
        coupling_parameters["hydrogen_storage_generation"] = [
            calculate_hydrogen_storage_generation(n)
        ]
        
        # Calculate battery storage
        coupling_parameters["battery_storage_generation"] = [
            calculate_battery_storage_generation(n)
        ]

        # Calculate generation shares
        coupling_parameters["generation_shares"] = [
            calculate_generation_shares(
                n,
                grouper=["region", "general_carrier"]
                )
        ]

        # Nested loop over perturbed networks if any
        for p in df.query("perturbed")["ptech"]:
            
            # Load perturbed network
            npert = pypsa.Network(df.query(f"ptech == '{p}'")["filepath"].values[0])
            
            # Add region and general_carrier to network components
            add_region_and_general_carrier(npert, region_mapping, map_pypsaeur_to_general)
            check_for_mapping_completeness(npert)
            
            # Partial derivative of capacity factors
            coupling_parameters["capacity_factors_derivative"] = difference_quotient(
                n_opt=n,
                n_pert=npert,
                ptech=p,
                property_func=calculate_capacity_factors,
                comps=["Generator"],
                grouper=["region", "general_carrier"],
                )
                    
            # Partial derivative of market values (supply side)
            coupling_parameters["markups_supply_derivative"] = difference_quotient(
                n_opt=n,
                n_pert=npert,
                ptech=p,
                property_func=calculate_markups_supply,
                comps=["Generator"],
                grouper=["region", "general_carrier"],
                )
    
    # Add year column to each coupling parameter
    for key, value in coupling_parameters.items():
        for df in value:
            df["year"] = year

    
    for fp in networks:
        # Extract year from filename, format: elec_y<YYYY>_<morestuff>.nc
        m = re.findall(r"y(\d{4})", fp)
        assert len(m) == 1, "Unable to extract year from network path"
        year = int(m[0])
        logger.info(f"Reading network for year: {year}")

        # Load network
        network = pypsa.Network(fp)

        # Check if network has objective attribute, if not: Optimisation most probably failed or something else went wrong. Raise an exception to notify loudly and stop the workflow
        if not hasattr(network, "objective"):
            raise ValueError(
                f"Network {fp} missing objective attribute, something probably went wrong in solving process during network optimisation."
            )

        # Add region to buses if it doesnt exist (this is the case if additionakl h2demand is not enabled)
        if "region" not in network.buses.columns:
            network.buses["region"] = ""

        # First map the PyPSA-EUR countries to REMIND-EU regions;
        # .statistics(..) can then automatically take care of the aggregation
        # H2 demand buses already have a region assigned, so we don't want to overwrite those
        network.buses["region"] = network.buses["region"].where(
            network.buses["region"] != "",
            network.buses["country"].map(region_mapping["REMIND-EU"]),
        )

        # Add information for aggregation later: region name (REMIND-EU) and general technology
        network.generators["region"] = network.generators["bus"].map(
            network.buses["region"]
        )
        network.stores["region"] = network.stores["bus"].map(network.buses["region"])
        network.storage_units["region"] = network.storage_units["bus"].map(
            network.buses["region"]
        )
        network.links["region"] = network.links["bus0"].map(network.buses["region"])
        network.lines["region"] = network.lines["bus0"].map(network.buses["region"])
        network.loads["region"] = network.loads["bus"].map(network.buses["region"])
        # Links/lines have two buses, and can be attributed to two regions (used for e.g. grid length calculations)
        network.links["region1"] = network.links["bus1"].map(network.buses["region"])
        network.lines["region1"] = network.lines["bus1"].map(network.buses["region"])

        network.generators["general_carrier"] = network.generators["carrier"].map(
            map_pypsaeur_to_general
        )
        network.stores["general_carrier"] = network.stores["carrier"]
        network.storage_units["general_carrier"] = network.storage_units["carrier"].map(
            map_pypsaeur_to_general
        )
        network.links["general_carrier"] = network.links["carrier"]
        network.lines["general_carrier"] = network.lines["carrier"]
        network.loads["general_carrier"] = network.loads["bus"].map(
            network.buses["carrier"]
        )

        # For separating generators which are RCL and those which are not (capacities reported separately)
        network.generators["RCL"] = False
        network.generators.loc[
            network.generators.index.str.contains("RCL"), "RCL"
        ] = True
        
        # For separating stores which are RCL and those which are not (capacities reported separately)
        network.links["RCL"] = False
        network.links.loc[
            network.links.index.str.contains("RCL"), "RCL"
        ] = True
        
        # For separating stores which are RCL and those which are not (capacities reported separately)
        network.stores["RCL"] = False
        network.stores.loc[
            network.stores.index.str.contains("RCL"), "RCL"
        ] = True

        # Hack: "hydro" representing hydro dams should be included in capacity and capacity factor calculations
        # By turning them into fake generators and setting the relevant attributes, they are included in the calculations by .statistics(..)
        # TODO: Check compatibility with statistics.energy_balance(..) and other statistics functions
        network.generators = pd.concat(
            [
                network.generators,
                network.storage_units.query(
                    "index.str.contains('hydro')", engine="python"
                ),
            ]
        )
        network.generators_t["p"] = network.pnl("Generator")["p"].join(
            network.pnl("StorageUnit")["p"].filter(like="hydro", axis="columns")
        )

        # Now make sure we have all carriers in the mapping
        check_for_mapping_completeness(network)

        ## Extract coupling parameters from network

        # Calculate capacity factors; by assigning the general carrier and grouping by here, the capacity factor is automatically
        # calculated across all "carrier" technologies that map to the same "general_carrier"
        capacity_factor = network.statistics.capacity_factor(
            comps=["Generator"], groupby=["region", "general_carrier"]
        )
        capacity_factor = (
            capacity_factor.to_frame("value").reset_index().drop(columns=["component"])
        )
        capacity_factor["year"] = year
        capacity_factors.append(capacity_factor)
        
        # Extract capacity factors of link components (H2 and batteries)
        capacity_factor_link = network.statistics.capacity_factor(
            comps=["Link"], groupby=["region", "general_carrier"]
        )
        capacity_factor_link = (
            capacity_factor_link.to_frame("value").reset_index().drop(columns=["component"])
        )
        capacity_factor_link["year"] = year
        # Remove H2 transfer to H2 demand REMIND (if additional H2 demand is on) and DC
        capacity_factor_link = capacity_factor_link.query("general_carrier != 'H2 transfer to H2 demand REMIND'")
        capacity_factor_link = capacity_factor_link.query("general_carrier != 'DC'")
        capacity_factors_links.append(capacity_factor_link)

        # Calculate availability factors
        availability_factor = calculate_availability_factor(
            network, comps=["Generator"], groupby=["region", "general_carrier"]
        )
        availability_factor = (
            availability_factor.to_frame("value")
            .reset_index()
            .drop(columns=["component"])
        )
        availability_factor["year"] = year
        availability_factors.append(availability_factor)

        # Calculate curtailment
        curtailment = network.statistics.curtailment(
            comps=["Generator"], groupby=["region", "general_carrier"]
        )
        curtailment = (
            curtailment.to_frame("value").reset_index().drop(columns=["component"])
        )
        curtailment["year"] = year
        curtailments.append(curtailment)

        # Calculate energy balance
        energy_balance = network.statistics.energy_balance(
            groupby=["region", "general_carrier"],
            bus_carrier = "AC"
        )
        energy_balance = energy_balance.to_frame("value").reset_index()
        energy_balance["year"] = year
        energy_balances.append(energy_balance)

        # Calculate shares of technologies in annual generation
        generation_share = (
            network.statistics.supply(
                comps=["Generator"],
                groupby=["region", "general_carrier"],
            )
            / network.statistics.supply(
                comps=["Generator"],
                groupby=["region", "general_carrier"],
            )
            .groupby(["region"])
            .sum()
        )
        generation_share = generation_share.to_frame("value").reset_index().drop(columns=["component"])
        generation_share["year"] = year
        generation_shares.append(generation_share)

        # Preinstalled capacities consist of two parts 
        # First, get p_nom from capacity-adjusted existing powerplants
        preinstalled_capacity_ppl = network.statistics.installed_capacity(
            comps=["Generator"], groupby=["RCL", "region", "general_carrier"]  # Only generators for now
        )
        preinstalled_capacity_ppl = preinstalled_capacity_ppl.to_frame("value").reset_index()
        # Second, get p_nom_opt from RCL components
        preinstalled_capacity_rcl = network.statistics.optimal_capacity(
            comps=["Generator", "Link", "Store"], groupby=["RCL", "region", "general_carrier"]
        )
        preinstalled_capacity_rcl = preinstalled_capacity_rcl.to_frame("value").reset_index()
        preinstalled_capacity_rcl = preinstalled_capacity_rcl.query("RCL == True")
        # Combine both
        preinstalled_capacity = pd.concat([preinstalled_capacity_ppl, preinstalled_capacity_rcl])
        preinstalled_capacity["year"] = year
        preinstalled_capacities.append(preinstalled_capacity)

        # Maximum potentials per technology
        # * RCL generators ignored, they are added with infinite capacity (instead limited with constraint)
        # * regular generators with p_nom_extendable == False by default only have p_nom assigned, which is used as potential
        df = network.generators.copy(deep=True)
        df = df.query("not index.str.contains('RCL')", engine="python")
        df["potential"] = df["p_nom"].where(
            df["p_nom_extendable"] == False, df["p_nom_max"]
        )
        potential = df.groupby(["region", "general_carrier"])["potential"].sum()
        # Drop infinities
        potential = potential.replace([np.inf, -np.inf], np.nan).dropna()
        potential = potential.to_frame("value").reset_index()
        potential["year"] = year
        potentials.append(potential)

        # Hourly prices per load type (eg. electricity, H2 buses) per region
        hourly_price = network.statistics.revenue(
            comps=["Load"],
            groupby=["region", "general_carrier"],
            aggregate_time=False,
        ) / (
            -1
            * network.statistics.withdrawal(
                comps=["Load"],
                groupby=["region", "general_carrier"],
                aggregate_time=False,
            )
        )
        hourly_price = hourly_price.reset_index()
        hourly_price["year"] = year
        hourly_prices.append(hourly_price)

        # Calculate load-weighted electricity prices based on bus marginal prices
        load_price = network.statistics.revenue(
            comps=["Load"], groupby=["region", "general_carrier"]
        ) / (
            -1
            * network.statistics.withdrawal(
                comps=["Load"], groupby=["region", "general_carrier"]
            )
        )
        load_price = (
            load_price.to_frame("value").reset_index().drop(columns=["component"])
        )
        load_price["year"] = year
        load_prices.append(load_price)

        ## Calculate load-weighted electricity price that all electrolysis sees
        # This uses the load of the electrolysis links as weightings
        # The previous implementaton that used the load of the "transfer to H2 demand REMIND"
        # did not make sense as the link p0 was arbitrary
        # (probably due to the standard H2 store at the buses)
        weightings = (
            network.links_t["p0"]
            .filter(regex="Electrolysis")
            .mul(network.snapshot_weightings["objective"], axis="rows")
        )
        weightings = weightings.rename(
            columns=network.links["bus0"]
        )
        # Add values if columns have the same name
        weightings = weightings.T.groupby(level=0).sum().T

       # Calculate electricity price paid by electrolysis
        electricity_price_electrolysis = (
            network.buses_t["marginal_price"][weightings.columns] * weightings
        ).rename(columns=network.buses["region"]).sum().groupby(
            level=0
        ).sum() / (weightings.rename(
            columns=network.buses["region"]
        ).sum().groupby(
            level=0
        ).sum())
        # Format data and append
        electricity_price_electrolysis = (
            electricity_price_electrolysis.to_frame("value")
            .reset_index()
            .rename(columns={"Bus": "region"})
        )
        electricity_price_electrolysis[
            "general_carrier"
        ] = "electricity price electrolysis"
        electricity_price_electrolysis["year"] = year
        electricity_prices_electrolysis.append(electricity_price_electrolysis)
        
        # TODO: Simpler way
        # gen = network.links_t.p0.loc[:, network.links.carrier == "H2 electrolysis"]
        # gen.columns = gen.columns.map(network.links.bus0)
        # gen = gen.T.groupby(level=0).sum().T
        # lmp = network.buses_t.marginal_price.loc[:,gen.columns]
        
        # mv = (gen * lmp).sum().sum() / gen.sum().sum()
        
        # Calculate loads (electricity and additional hydrogen demand) per region
        load = network.statistics.withdrawal(
            comps=["Load"], groupby=["region", "general_carrier"]
        )
        load = load.to_frame("value").reset_index()
        load["year"] = year
        loads.append(load)

        # Calculate crossborder flows and prices
        crossborder_flow, crossborder_price_import, crossborder_price_export = determine_crossborder_flow_and_price(
            network, carrier=["AC", "DC"]
        )
        crossborder_flow["year"] = year
        crossborder_price_import["year"] = year
        crossborder_price_export["year"] = year
        crossborder_flows.append(crossborder_flow)
        crossborder_prices_import.append(crossborder_price_import)
        crossborder_prices_export.append(crossborder_price_export)

        # Calculate share of the generation in each region in total generation
        # This is used to parametrise the pre-factor equation for electricity trade in REMIND 
        generation_region_share = (
            network.statistics.supply(comps=["Generator"], groupby=["region"])
            .to_frame("value")
            .apply(lambda x: x/sum(x))
            )

        generation_region_share = generation_region_share.reset_index().drop(columns=["component"])
        generation_region_share["year"] = year
        generation_region_shares.append(generation_region_share)

        ## Calculate optimal capacities
        optimal_capacity = network.statistics.optimal_capacity(
            comps=["Generator", "Load", "Link", "Line", "Store", "StorageUnit"],
            groupby=["region", "general_carrier"],
        )
        optimal_capacity = optimal_capacity.to_frame("value").reset_index()
        optimal_capacity["year"] = year
        optimal_capacities.append(optimal_capacity)

        ## Calculate peak residual load
        dispatchable_technologies = set(network.generators.index) - set(
            network.generators_t.p_max_pu.columns
        )

        # Add attribute to network.generators to distinguish between dispatchable and non-dispatchable technologies
        network.generators["peak_residual_load"] = "No"
        network.generators.loc[
            list(dispatchable_technologies), "peak_residual_load"
        ] = "Yes"
        # Don't include load shedding as dispatchable technology
        network.generators.loc[network.generators.index.str.contains("load"), "peak_residual_load"] = "No"
        network.loads["peak_residual_load"] = "Load"
        # Don't include hydrogen turbines and batteries into peak residual load calculation
        network.stores["peak_residual_load"] = "No"
        # Don't include hydro and pumped hydro into peak residual load calculation (no PHS in REMIND)
        network.storage_units["peak_residual_load"] = "No"

        residual_load = (
            network.statistics.energy_balance(
                comps=["Generator", "Store", "StorageUnit", "Load"],
                bus_carrier="AC",
                groupby=["region", "peak_residual_load"],
                aggregate_time=False,
            )
            .groupby(["region", "peak_residual_load"])
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
                    "absolute": x.xs("Yes", level="peak_residual_load")[
                        max_prl_snapshot
                    ]
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
            residual_load.groupby("region")
            .apply(get_absolute_and_relative_prl)
            .reset_index()
        )
        peak_residual_load["year"] = year
        peak_residual_load["carrier"] = "AC"
        peak_residual_loads.append(peak_residual_load)

        def get_supply_with_zeros(value):
            supply_data = network.statistics.supply(comps=["Link"], groupby=["region", "carrier"])
            # Ensure the index includes both 'component' and 'region'
            supply_data = supply_data.xs("Link", level="component")  # Extract only "Link" component

            # Convert to DataFrame and unstack "carrier"
            supply_df = supply_data.unstack(level="carrier").fillna(0)

            # Ensure "battery discharger" column exists
            if value not in supply_df.columns:
                supply_df[value] = 0

            # Extract the "battery discharger" supply and restore MultiIndex
            absolute_supply = supply_df[value]
            absolute_supply = absolute_supply.to_frame("objective")  # Convert to DataFrame

            # Reintroduce "component" level
            absolute_supply["component"] = "Link"
            absolute_supply = absolute_supply.set_index("component", append=True)  # Move "component" to MultiIndex
            absolute_supply = absolute_supply.reorder_levels(["component", "region"])["objective"]  # Ensure correct order
            
            return absolute_supply

        ## Calculate hydrogen turbine (fuel cell) storage requirements
        absolute_supply = get_supply_with_zeros("H2 fuel cell")
        relative_supply = absolute_supply / network.statistics.withdrawal(comps="Load", bus_carrier="AC", groupby="region")

        h2turb_storage = pd.DataFrame({
            "absolute": absolute_supply,
            "relative": relative_supply
        }).reset_index()
        h2turb_storage["year"] = year
        h2turb_storage["carrier"] = "H2 fuel cell"
        h2turb_storages.append(h2turb_storage)
        
        ## Calculate battery storage requirements
        absolute_supply = get_supply_with_zeros("battery discharger")
        relative_supply = absolute_supply / network.statistics.withdrawal(comps="Load", bus_carrier="AC", groupby="region")
        
        battery_storage = pd.DataFrame({
            "absolute": absolute_supply,
            "relative": relative_supply
        }).reset_index()
        battery_storage["year"] = year
        battery_storage["carrier"] = "battery discharger"
        battery_storages.append(battery_storage)
        
        ## Determine grid losses in absolute and relative terms
        grid_loss_abs = network.statistics.energy_balance(comps = "Line", groupby="region").abs()
        # TODO: Hacky temporary solution
        if len(grid_loss_abs) == 0:
            grid_loss_abs = pd.Series([0], index=["DEU"])
            grid_loss_abs.index.name = "region"
        grid_loss_rel = grid_loss_abs / network.statistics.withdrawal(comps="Load", bus_carrier="AC", groupby="region")
        
        grid_loss = pd.DataFrame({
            "absolute": grid_loss_abs,
            "relative": grid_loss_rel
        }).reset_index()
        grid_loss["year"] = year
        grid_loss["carrier"] = "AC"
        grid_losses.append(grid_loss)

        ## Determime grid sizes and investments per region
        # The full grid: Combine DC and AC lines into one dataframe
        # TODO: Refactor when implementing grid tech in REMIND
        grid = pd.concat(
            [
                network.links.query("carrier == 'DC'")[
                    [
                        "p_nom_opt",
                        "length",
                        "region",
                        "region1",
                        "carrier",
                        "capital_cost",
                    ]
                ],
                network.lines[
                    [
                        "s_nom_opt",
                        "length",
                        "region",
                        "region1",
                        "carrier",
                        "capital_cost",
                    ]
                ].rename(columns={"s_nom_opt": "p_nom_opt"}),
            ]
        )

        # Separate national and international grid, as the international grid requires further preprocessing
        national_grid = grid.where(grid["region"] == grid["region1"]).dropna()
        international_grid = grid.where(grid["region"] != grid["region1"]).dropna()

        # National grid: "region" is left unchanged; same region for both buses
        # International grid: assigning to a region is more difficult, as the grid is connected to two regions,
        # solution: duplicate the connections and assign half the length to each region
        international_grid["length"] /= 2
        new_region = pd.concat(
            [international_grid["region"], international_grid["region1"]]
        )
        international_grid = pd.concat([international_grid] * 2)
        international_grid["region"] = new_region.values
        international_grid.index = (
            international_grid.index + international_grid["region"]
        )

        # Recombine national and international grid
        grid = pd.concat([national_grid, international_grid])
        
        # If there are no grid connections, create data frame with only zeros
        # TODO: Hacky way to avoid errors
        if len(grid) == 0:
            grid = pd.DataFrame({
                "region": ["DEU"],
                "p_nom_opt": [0],
                "length": [0],
                "carrier": ["AC-DC"],
                "capital_cost": [0]
            })
        # calculate total grid capacity in (MW*km) per region
        grid_capacity = (
            grid.groupby(["region"])
            .apply(lambda x: (x["p_nom_opt"] * x["length"]).sum(), include_groups=False)
            .to_frame("value")
            .reset_index()
        )
        grid_capacity["year"] = year
        grid_capacity["carrier"] = "AC-DC"
        grid_capacities.append(grid_capacity)

        grid_investment = (
            grid.groupby(["region"])
            .apply(lambda x: (x["p_nom_opt"] * x["capital_cost"]).sum(), include_groups=False)
            .to_frame("value")
            .reset_index()
        )
        # Hack: Convert annualised investment costs to total investment costs; "17" corresponds to the approximate annuity factor for 30 years at 5% interest rate
        # TODO:
        # Implement transmission technology in REMIND and couple REMIND investment costs to PyPSA-Eur.
        # This would allow us to avoid passing the investment costs from PyPSA-Eur to REMIND alltogether.
        grid_investment["value"] *= 17
        grid_investment["year"] = year
        grid_investment["carrier"] = "AC-DC"
        grid_investments.append(grid_investment)

        # Calculate market values after applying cutoff for electricity prices
        if cutoff_market_values := snakemake.config["remind_coupling"][
            "extract_coupling_parameters"
        ]["cutoff_market_values"]:
            relevant_buses = network.buses.query("carrier == 'AC'").index
            cutoff_market_values = float(cutoff_market_values)
            zscores = (
                network.buses_t["marginal_price"][relevant_buses]
                .apply(zscore)
                .mean(axis="columns")
            )
            zscores.index = pd.to_datetime(zscores.index)

            # By setting snapshot_weightings to 0, the market value will not be calculated for these snapshots above the cutoff value
            network.snapshot_weightings = network.snapshot_weightings.where(
                zscores < cutoff_market_values, 0
            )

            logger.info(
                "Cutoff for electricity prices in market value calculation enabled. "
                "Excluding {n} snapshots from calculations with electricity prices above {p:.2f} USD/MWh.".format(
                    n=int(
                        network.snapshot_weightings["generators"].shape[0]
                        * network.snapshot_weightings["generators"].iloc[0]
                        - network.snapshot_weightings["generators"].sum()
                    ),
                    p=network.buses_t["marginal_price"][relevant_buses]
                    .where(zscores < cutoff_market_values)
                    .mean(axis="columns")
                    .max(),
                )
            )

        # Calculate the market values (round-about way as the intended method of the statistics module is not yet available)
        market_value = network.statistics.market_value(
            comps=["Generator"],
            groupby=["region", "general_carrier"],
        )
        market_value = (
            market_value.to_frame("value").reset_index().drop(columns=["component"])
        )
        market_value["year"] = year
        market_values.append(market_value)
        
        # Calculate markup = market value - load price (AC carrier)
        markup = market_value.drop(columns=["year"]).merge(
            load_price.query("general_carrier == 'AC'").drop(columns=["general_carrier", "year"]),
            on=["region"])
        markup["value"] = markup["value_x"] - markup["value_y"]
        markup = markup.drop(columns=["value_x", "value_y"])
        # Replace NaNs with 0.01 $/MWh (arbitrary small value), passing zero causes problems in GAMS
        markup = markup.fillna(0.01)
        markup["year"] = year
        markups.append(markup)

    
    def weigh_by_REMIND_capacity(df):
        """
        Weighing here uses the capacities from REMIND and calaculates the
        weights s.t.

        the sum of weights equals 1 for each group of carriers (=
        "general_carrier") which are mapped against REMIND technologies.
        """
        # Read gen shares from REMIND for weighing
        capacity_weights = read_remind_data(
            file_path=snakemake.input["remind_weights"],
            variable_name="v32_shPe2seel",
            rename_columns={
                "ttot": "year",
                "all_regi": "region",
                "all_te": "carrier",
            },
        )

        # Align dtypes & reduce to required columns
        capacity_weights = capacity_weights.astype(
            {"year": int, "level": float, "carrier": str, "region": str}
        )
        capacity_weights = capacity_weights[["year", "region", "carrier", "level"]]
        capacity_weights["level"] = capacity_weights["level"].where(
            lambda x: x > np.finfo(float).eps, 0.0
        )  # Remove very small values below EPS passed by GAMS

        # Map REMIND technologies to general_carriers
        capacity_weights["general_carrier"] = capacity_weights["carrier"].map(
            {lv: k for k, v in map_general_to_remind.items() for lv in v}
        )

        # Calculate weights for individual carrier based on total levels per shared "general_carrier" and individual share
        general_carrier_weights = capacity_weights.groupby(
            ["year", "region", "general_carrier"]
        )["level"].sum()
        general_carrier_weights = general_carrier_weights.where(
            lambda x: x != 0, 1
        )  # avoid division by zero
        general_carrier_weights.name = "general_carrier_weight"

        capacity_weights = capacity_weights.join(
            general_carrier_weights,
            on=["year", "region", "general_carrier"],
            validate="m:1",
        )
        capacity_weights["weight"] = (
            capacity_weights["level"] / capacity_weights["general_carrier_weight"]
        )

        # Map weights to data to-be-weighted based on (year, region, carrier)
        df = df.set_index(["year", "region", "carrier"]).join(
            capacity_weights.set_index(["year", "region", "carrier"])
        )

        # Consistency checks
        assert df["weight"].isna().sum() == 0, "Some weights are missing"
        assert all(df["weight"] >= 0.0), "Some weights are negative"
        assert all(df["weight"] <= 1.0), "Some weights are larger than 1.0"

        # Apply weights
        df["value"] *= df["weight"]

        return df[["value"]].reset_index()

    # %%
    # Real combining happens here
    capacity_factors = postprocess_dataframe(capacity_factors)
    capacity_factors_links = postprocess_dataframe(capacity_factors_links, map_to_remind=False)
    availability_factors = postprocess_dataframe(availability_factors)
    curtailments = postprocess_dataframe(curtailments)
    generation_shares = postprocess_dataframe(generation_shares)
    peak_residual_loads = postprocess_dataframe(
        peak_residual_loads, map_to_remind=False
    )
    grid_losses = postprocess_dataframe(grid_losses, map_to_remind=False)
    grid_capacities = postprocess_dataframe(grid_capacities, map_to_remind=False)
    grid_investments = postprocess_dataframe(grid_investments, map_to_remind=False)
    market_values = postprocess_dataframe(market_values)
    hourly_prices = postprocess_dataframe(hourly_prices)
    load_prices = postprocess_dataframe(load_prices)
    markups = postprocess_dataframe(markups)
    electricity_prices_electrolysis = postprocess_dataframe(
        electricity_prices_electrolysis, map_to_remind=False
    )
    loads = postprocess_dataframe(loads)
    potentials = postprocess_dataframe(potentials, map_to_remind=True)
    h2turb_storages = postprocess_dataframe(h2turb_storages, map_to_remind=False)
    battery_storages = postprocess_dataframe(battery_storages, map_to_remind=False)
    # Only reporting for plotting, not coupled, therefore other treatment
    preinstalled_capacities = postprocess_dataframe(
        preinstalled_capacities, map_to_remind=False
    )
    energy_balances = postprocess_dataframe(energy_balances, map_to_remind=False)

    optimal_capacities = (
        pd.concat(optimal_capacities)
        .rename(columns={"component": "type", "general_carrier": "carrier"})
        .set_index(["year", "region", "type", "carrier"])
        .sort_index()["value"]
        .reset_index()
    )

    crossborder_flows = (
        pd.concat(crossborder_flows)
        .set_index(["year", "from", "to"])
        .sort_index()
        .reset_index()
    )

    crossborder_prices_import = (
        pd.concat(crossborder_prices_import)
        .set_index(["year", "from", "to"])
        .sort_index()
        .reset_index()
    )

    crossborder_prices_export = (
        pd.concat(crossborder_prices_export)
        .set_index(["year", "from", "to"])
        .sort_index()
        .reset_index()
    )

    generation_region_shares = (
        pd.concat(generation_region_shares)
        .set_index(["year", "region"])
        .sort_index()
        .reset_index()
    )

    # TODO: Remove
    peak_residual_loads = peak_residual_loads.query("carrier == 'AC'").drop(
        columns=["carrier"]
    )

    # %%
    # Special treatment: Weigh values of df based on installed capacities in REMIND
    generation_shares = weigh_by_REMIND_capacity(generation_shares)

    # Throw warning if sum is not between 0.99 and 1.01 for each year and region (allowing for some numerical tolerance)
    if any(
        generation_shares.groupby(["year", "region"])["value"]
        .sum()
        .where(lambda x: (x < 0.99) | (x > 1.01))
        .notna()
    ):
        logger.warning(
            "Sum of generation shares is not between 0.99 and 1.01 for each year and region:"
        )
        logger.warning(
            generation_shares.groupby(["year", "region"])["value"].sum().where(
                lambda x: (x < 0.99) | (x > 1.01)
            ).dropna()
        )
        

    # %%
    # Export as csv values (informative purposes only, coupling parameters below via GDX)
    for fn, df in {
        "capacity_factors": capacity_factors,
        "capacity_factors_links": capacity_factors_links,
        "availability_factors": availability_factors,
        "curtailments": curtailments,
        "generation_shares": generation_shares,
        "peak_residual_loads": peak_residual_loads,
        "grid_losses": grid_losses,
        "grid_capacities": grid_capacities,
        "grid_investments": grid_investments,
        "preinstalled_capacities": preinstalled_capacities,
        "market_values": market_values,
        "hourly_prices": hourly_prices,
        "load_prices": load_prices,
        "markups": markups,
        "electricity_prices_electrolysis": electricity_prices_electrolysis,
        "loads": loads,
        "energy_balances": energy_balances,
        "potentials": potentials,
        "optimal_capacities": optimal_capacities,
        "h2turb_storages": h2turb_storages,
        "battery_storages": battery_storages,
        "crossborder_flows": crossborder_flows,
        "crossborder_prices_import": crossborder_prices_import,
        "crossborder_prices_export": crossborder_prices_export,
        "generation_region_shares": generation_region_shares,
    }.items():
        df.to_csv(snakemake.output[fn], index=False)

    # %%
    # Export to GAMS gdx file for coupling
    gdx = gt.Container()

    # Construct sets from exemplary index; luckily all data share most of the indices
    sets = {
        "year": market_values["year"].unique(),
        "region": market_values["region"].unique(),
        "carrier": np.append(
            market_values["carrier"].unique(), load_prices["carrier"].unique()
        ),
        "storage_and_transmission_technologies": [
            "AC",
            "DC",
            "H2",
            "H2 fuel cell",
            "H2 electrolysis",
            "battery",
            "battery charger",
            "battery discharger",
        ],
        "grid_technologies": ["AC-DC"],
        "electrolysis": ["electricity price electrolysis"],
    }

    # First add sets to the container
    s_year = gt.Set(gdx, "year", records=sets["year"], description="simulation year")
    s_region = gt.Set(
        gdx,
        "region",
        records=sets["region"],
        description="REMIND-EU region by which the values were aggregated",
    )
    s_carrier = gt.Set(
        gdx,
        "carrier",
        records=sets["carrier"],
        description="PyPSA technology by which the values were aggregated",
    )
    s_storage_and_transmission_technologies = gt.Set(
        gdx,
        "storage_and_transmission_technologies",
        records=sets["storage_and_transmission_technologies"],
        description="Storage and transmission technologies exported from PyPSAEur",
    )
    s_grid_technologies = gt.Set(
        gdx,
        "grid_technologies",
        records=sets["grid_technologies"],
        description="Grid technologies exported from PyPSAEur",
    )

    # Now we can add data to the container
    c = gt.Parameter(
        gdx,
        name="capacity_factor",
        domain=[s_year, s_region, s_carrier],
        records=capacity_factors,
        description="Cacacity factors of technology per year and region in p.u.",
    )
    
    cl = gt.Parameter(
        gdx,
        name="storage_and_transmission_capacity_factors",
        domain=[s_year, s_region, s_storage_and_transmission_technologies],
        records=capacity_factors_links,
        description="Capacity factors of storage and transmission technologies per year and region in p.u.",
    )

    a = gt.Parameter(
        gdx,
        name="availability_factor",
        domain=[s_year, s_region, s_carrier],
        records=availability_factors,
        description="Availability factors of technology per year and region in p.u.",
    )

    cu = gt.Parameter(
        gdx,
        name="curtailment",
        domain=[s_year, s_region, s_carrier],
        records=curtailments,
        description="Curtailment of technology per year and region in MWh",
    )

    g = gt.Parameter(
        gdx,
        name="generation_share",
        domain=[s_year, s_region, s_carrier],
        records=generation_shares,
        description="Share of generation of technology per year and region in p.u.",
    )

    prla = gt.Parameter(
        gdx,
        name="peak_residual_load_absolute",
        domain=[s_year, s_region],
        records=peak_residual_loads[["year", "region", "absolute"]],
        description="Peak residual load per year and region as absolute value in MWh",
    )

    prlr = gt.Parameter(
        gdx,
        name="peak_residual_load_relative",
        domain=[s_year, s_region],
        records=peak_residual_loads[["year", "region", "relative"]],
        description="Peak residual load per year and region relative to mean load in p.u.",
    )
    
    h2sr = gt.Parameter(
        gdx,
        name="h2turb_storage_relative",
        domain=[s_year, s_region],
        records=h2turb_storages[["year", "region", "relative"]],
        description="Hydrogen turbine supply per year and region relative to total load in p.u.",
    )
    
    btsr = gt.Parameter(
        gdx,
        name="battery_storage_relative",
        domain=[s_year, s_region],
        records=battery_storages[["year", "region", "relative"]],
        description="Battery discharging per year and region relative to total load in p.u.",
    )

    xbf = gt.Parameter(
        gdx,
        name="crossborder_flow",
        domain=[s_year, s_region, s_region],
        records=crossborder_flows,
        description="Crossborder flows from region (columnn 2) to region (column 3) per year in MWh",
    )

    xbpi = gt.Parameter(
        gdx,
        name="crossborder_price_import",
        domain=[s_year, s_region, s_region],
        records=crossborder_prices_import,
        description="Crossborder prices paid by the importing region (column 3) from trade with the exporting region (column 2) per year in EUR/MWh",
    )

    xbpe = gt.Parameter(
        gdx,
        name="crossborder_price_export",
        domain=[s_year, s_region, s_region],
        records=crossborder_prices_export,
        description="Crossborder prices received by the exporting region (column 2) from trade with the importing region (column 3) per year in EUR/MWh",
    )

    grs = gt.Parameter(
        gdx,
        name="generation_region_share",
        domain=[s_year, s_region],
        records=generation_region_shares,
        description="Share of generation of region in total generation per year in p.u.",
    )
    
    glr = gt.Parameter(
        gdx,
        name="grid_loss_relative",
        domain=[s_year, s_region],
        records=grid_losses[["year", "region", "relative"]],
        description="Grid losses per year and region relative to total load in p.u.",
    )

    gc = gt.Parameter(
        gdx,
        name="grid_capacity",
        domain=[s_year, s_region, s_grid_technologies],
        records=grid_capacities,
        description="AC and DC Grid capacity per year and region in (MW*km)",
    )

    gi = gt.Parameter(
        gdx,
        name="grid_investment",
        domain=[s_year, s_region, s_grid_technologies],
        records=grid_investments,
        description="AC and DC Grid investment (estimated!) per year and region in USD. "
        "Reminder to implement transmission technology in REMIND and couple REMIND investment costs to PyPSA-Eur. "
        "Then remove this coupled parameter and use the REMIND investment costs instead.",
    )

    m = gt.Parameter(
        gdx,
        name="market_value",
        domain=[s_year, s_region, s_carrier],
        records=market_values,
        description="Market value of technology per year and region in EUR/MWh",
    )

    p = gt.Parameter(
        gdx,
        name="load_price",
        domain=[s_year, s_region, s_carrier],
        records=load_prices,
        description="Prices for load types (electricity, hydrogen) per year and region in EUR/MWh (electricity, hydrogen LHV)",
    )
    
    mu = gt.Parameter(
        gdx,
        name="markup",
        domain=[s_year, s_region, s_carrier],
        records=markups,
        description="Markup = market value minus load price per year and region in EUR/MWh",
    )

    oc_st = gt.Parameter(
        gdx,
        name="storage_and_transmission_capacities",
        domain=[s_year, s_region, s_storage_and_transmission_technologies],
        # restrict output here to selected technologies and slice for only the relevant columns
        records=optimal_capacities.loc[
            optimal_capacities.carrier.isin(
                sets["storage_and_transmission_technologies"]
            )
        ][["year", "region", "carrier", "value"]],
        description="Optimal capacity per year and region in MW (generators, links, lines) or MWh (stores)",
    )

    epe = gt.Parameter(
        gdx,
        name="electricity_price_electrolysis",
        domain=[s_year, s_region],
        records=electricity_prices_electrolysis[["year", "region", "value"]],
        description="Electricity price paid by electrolysis per year and region in [EUR/MWh].",
    )

    pot = gt.Parameter(
        gdx,
        name="potential",
        domain=[s_year, s_region, s_carrier],
        records=potentials,
        description="Maximum potential of technology per year and region in MW",
    )

    gdx.write(snakemake.output["gdx"])

    # Export the technology mapping used
    logger.info(
        f"Exporting technology mapping from {snakemake.input['technology_cost_mapping']}"
    )
    get_technology_mapping(
        snakemake.input["technology_cost_mapping"], group_technologies=True
    ).to_csv(snakemake.output["technology_mapping"], index=False)

# %%
