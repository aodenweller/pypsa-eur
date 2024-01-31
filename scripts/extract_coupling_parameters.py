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


# Non-standard function, following standard implementation from PyPSA.statistics
def calculate_availability_factor(
    n,
    comps=None,
    aggregate_time="mean",
    aggregate_groups="sum",
    groupby=None,
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

    def func(n, c):
        p = get_availability(n, c).abs()
        weights = pypsa.statistics.get_weightings(n, c)
        return pypsa.statistics.aggregate_timeseries(p, weights, agg=aggregate_time)

    df = pypsa.statistics.aggregate_components(
        n, func, comps=comps, agg=aggregate_groups, groupby=groupby
    )

    capacity = n.statistics.optimal_capacity(
        comps=comps, aggregate_groups=aggregate_groups, groupby=groupby
    )
    df = df.div(capacity, axis=0)
    df.attrs["name"] = "Availability Factor"
    df.attrs["unit"] = "p.u."
    return df


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "extract_coupling_parameters",
            configfiles="config/config.remind.yaml",
            iteration="5",
            scenario="PyPSA_NPi_multiregion_2024-01-21_15.14.36",
        )

        # mock_snakemake doesn't work with checkpoints
        input_networks = [
            f"../results/{snakemake.wildcards['scenario']}/i{snakemake.wildcards['iteration']}/y{year}/networks/elec_s_6_ec_lcopt_3H-RCL-Ep{ep:.1f}.nc"
            for (year, ep) in zip(
                # pairs of years and ...
                [
                    2025,
                    2030,
                    2035,
                    2040,
                    2045,
                    2050,
                    2055,
                    2060,
                    2070,
                    2080,
                    2090,
                    2100,
                    2110,
                    2130,
                    2150,
                ],
                # ... emission prices (ep)
                [
                    25.2,
                    25.6,
                    26.4,
                    27.4,
                    28.8,
                    30.4,
                    32.4,
                    34.6,
                    40.0,
                    45.0,
                    50.0,
                    55.0,
                    60.0,
                    70.0,
                    80.0,
                ],
            )
        ]
    else:
        input_networks = snakemake.input["networks"]
        configure_logging(snakemake)

    # Only Generation technologies (PyPSA "generator" "carriers")
    # Use a two step mapping approach between PyPSA-EUR and REMIND:
    # First mapping is aggregating PyPSA-EUR technologies to general technologies
    # Second mapping is disaggregating general technologies to REMIND technologies
    map_pypsaeur_to_general = (
        get_technology_mapping(
            snakemake.input["technology_cost_mapping"], group_technologies=True
        )
        .groupby("PyPSA-Eur")
        .agg(lambda x: list(set(x))[0])["technology_group"]
        .to_dict()
    )
    map_pypsaeur_to_general.pop("offwind")  # not needed

    map_general_to_remind = (
        get_technology_mapping(
            snakemake.input["technology_cost_mapping"], group_technologies=True
        )
        .groupby("technology_group")
        .agg(lambda x: list(set(x)))["REMIND-EU"]
        .to_dict()
    )

    map_pypsaeur_to_remind_loads = {
        "AC": ["AC"],
        "H2 demand REMIND": ["H2 demand REMIND"],
    }

    # Create region mapping
    region_mapping = get_region_mapping(
        snakemake.input["region_mapping"], source="PyPSA-EUR", target="REMIND-EU"
    )
    region_mapping = pd.DataFrame(region_mapping).T.reset_index()
    region_mapping.columns = ["PyPSA-EUR", "REMIND-EU"]
    region_mapping = region_mapping.set_index("PyPSA-EUR")

    def check_for_mapping_completeness(n):
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
        price_import = network.buses_t["marginal_price"][p.columns.get_level_values("to")]  # Change "to" to use column_names?
        price_import.columns = p.columns

        # Calculate total expenditure for importing electricity
        expenditure_import = p.mul(price_import).T

        # Map buses to regions
        expenditure_import.index = (
            pd.MultiIndex.from_arrays([
                expenditure_import.index.get_level_values("from")
                .map(network.buses["region"]),
                expenditure_import.index.get_level_values("to")
                .map(network.buses["region"])
            ], names=["from", "to"])
        )

        expenditure_import = expenditure_import.groupby(["from", "to"]).sum().sum(axis="columns")

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
        price = expenditure_import / p

        # Convert to dataframe
        p = p.to_frame("exports").reset_index()
        price = price.to_frame("price").reset_index()

        return p, price

    capacity_factors = []
    availability_factors = []
    curtailments = []
    generation_shares = []
    generations = []
    preinstalled_capacities = []
    potentials = []
    peak_residual_loads = []
    grid_capacities = []
    grid_investments = []
    market_values = []
    load_prices = []
    electricity_prices_electrolysis = []
    hourly_prices = []
    crossborder_flows = []
    crossborder_prices = []
    generation_region_shares = []

    # Values used for reporting but not for coupling
    electricity_loads = []
    optimal_capacities = []

    for fp in input_networks:
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

        # Hack: "hydro" representing hydro dams should be included in capacity and capacity factor calculations
        # By turning them into fake generators and setting the relevant attributes, they are included in the calculations by .statistics(..)
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

        # Calculate generation
        generation = network.statistics.supply(
            comps=["Generator"], groupby=["region", "general_carrier"]
        )
        generation = generation.to_frame("value").reset_index()
        generation["year"] = year
        generations.append(generation)

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

        # Calculate technology pre-installed capacities
        # RCL-capacities are <= pre-installed capacities provided from REMIND, choose RCL capacities here
        # as starter; these are first expanded before the same carriers but non-RCL are installed (due to 0 costs)
        preinstalled_capacity = network.statistics.optimal_capacity(
            comps=["Generator"], groupby=["RCL", "region", "general_carrier"]
        )
        preinstalled_capacity = preinstalled_capacity.to_frame("value").reset_index()
        preinstalled_capacity["year"] = year
        preinstalled_capacity = preinstalled_capacity.query("RCL == True")
        preinstalled_capacities.append(preinstalled_capacity)

        # Maximum potentials per technology
        # * RCL generators ignored, they are added with infinite capacity (instead limited with constraint)
        # * regular generators with p_nom_extendable == False by default only have p_nom assigned, which is used as potential
        df = network.generators.copy(deep=True)
        df = df.query("not index.str.contains('RCL')", engine="python")
        df["potential"] = df["p_nom"].where(
            df["p_nom_extendable"] == False, df["p_nom_max"]
        )
        potential = df.groupby(["region", "general_carrier"])["potential"].apply(
            np.sum
        )  # np.sum: work-around pandas bug turnin inf to nan
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

        ## Calculate load-weighted electricity prices that H2 electrolysis sees for meeting
        # H2 demand from REMIND (implicit assumption: H2 is directly fed to regional H2 demand and not stored
        # in H2 cavern storage used by standard PyPSAEur for long-term electricity storage)

        # Weighting for electricity prices: when electrolysis (proxied by the transfer link) is active to meet
        # REMIND H2 demand. Renaming of column names to match electricity bus names
        weightings = (
            network.links_t["p0"]
            .filter(regex="transfer to [A-Z]{3} H2 demand REMIND")
            .mul(network.snapshot_weightings["objective"], axis="rows")
        )
        weightings = weightings.rename(
            columns=lambda x: re.search(
                r"(.*) H2 transfer to [A-Z]{3} H2 demand REMIND", x
            ).group(1)
        )
        # aggregation by country, but we want REMIND regions, so first need to map countries to regions before we aggregate
        electricity_price_electrolysis = (
            network.buses_t["marginal_price"][weightings.columns] * weightings
        ).rename(columns=network.buses["region"]).sum().groupby(
            level=0
        ).sum() / weightings.rename(
            columns=network.buses["region"]
        ).sum().groupby(
            level=0
        ).sum()
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
    
        # Calculate electricity loads per region
        electricity_load = network.statistics.withdrawal(
            comps=["Load"], groupby=["region", "general_carrier"]
        )
        electricity_load = electricity_load.to_frame("value").reset_index()
        electricity_load["year"] = year
        electricity_loads.append(electricity_load)

        # Calculate crossborder flows and prices
        crossborder_flow, crossborder_price = determine_crossborder_flow_and_price(
            network, carrier=["AC", "DC"]
        )
        crossborder_flow["year"] = year
        crossborder_price["year"] = year
        crossborder_flows.append(crossborder_flow)
        crossborder_prices.append(crossborder_price)

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
        network.loads["peak_residual_load"] = "Load"
        network.stores["peak_residual_load"] = "No"
        network.storage_units["peak_residual_load"] = "No"

        residual_load = (
            network.statistics.dispatch(
                comps=["Generator", "Store", "StorageUnit", "Load"],
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

        ## Determime grid sizes and investments per region
        # The full grid: Combine DC and AC lines into one dataframe
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

        # calculate total grid capacity in (MW*km) per region
        grid_capacity = (
            grid.groupby(["region"])
            .apply(lambda x: (x["p_nom_opt"] * x["length"]).sum())
            .to_frame("value")
            .reset_index()
        )
        grid_capacity["year"] = year
        grid_capacity["carrier"] = "AC-DC"
        grid_capacities.append(grid_capacity)

        grid_investment = (
            grid.groupby(["region"])
            .apply(lambda x: (x["p_nom_opt"] * x["capital_cost"]).sum())
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

    # %%
    ## Combine DataFrames to same format
    # Helper function
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

        df = df.set_index(
            ["year", "region", "carrier"]
        ).sort_index()  # set and sort by index for more logical sort order

        df = df.reset_index().drop(
            columns=["level_0"],
            errors="ignore",
        )  # Restructure and remove excess column

        def map_carriers_for_remind(dg):
            """
            Maps the carrier names from general technologies to REMIND
            technologies.
            """
            old_carrier = dg.iloc[0]["carrier"]

            # Mapping for generators and loads are different
            if old_carrier in map_general_to_remind.keys():
                _map = map_general_to_remind
            elif old_carrier in map_pypsaeur_to_remind_loads.keys():
                _map = map_pypsaeur_to_remind_loads

            new_carriers = _map[old_carrier]

            # Repeat rows for each new carrier, create new dataframe then assign the new carrier name
            dg = pd.DataFrame(
                np.repeat(dg.values, len(new_carriers), axis=0), columns=dg.columns
            )
            dg["carrier"] = new_carriers
            return dg

        if map_to_remind:
            # Map carriers to REMIND technologies
            df = df.groupby(["year", "region", "carrier"], group_keys=False).apply(
                map_carriers_for_remind
            )

        return df

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
            variable_name="v32_shSeElDisp",
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
    availability_factors = postprocess_dataframe(availability_factors)
    curtailments = postprocess_dataframe(curtailments)
    generation_shares = postprocess_dataframe(generation_shares)
    peak_residual_loads = postprocess_dataframe(
        peak_residual_loads, map_to_remind=False
    )
    grid_capacities = postprocess_dataframe(grid_capacities, map_to_remind=False)
    grid_investments = postprocess_dataframe(grid_investments, map_to_remind=False)
    market_values = postprocess_dataframe(market_values)
    hourly_prices = postprocess_dataframe(hourly_prices)
    load_prices = postprocess_dataframe(load_prices)
    electricity_prices_electrolysis = postprocess_dataframe(
        electricity_prices_electrolysis, map_to_remind=False
    )
    electricity_loads = postprocess_dataframe(electricity_loads)
    # Only reporting for plotting, not coupled, therefore other treatment
    preinstalled_capacities = postprocess_dataframe(
        preinstalled_capacities, map_to_remind=False
    )
    generations = postprocess_dataframe(generations, map_to_remind=False)
    potentials = postprocess_dataframe(potentials, map_to_remind=False)

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

    crossborder_prices = (
        pd.concat(crossborder_prices)
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

    electricity_loads = electricity_loads.query("carrier == 'AC'").drop(
        columns=["carrier"]
    )
    peak_residual_loads = peak_residual_loads.query("carrier == 'AC'").drop(
        columns=["carrier"]
    )

    # %%
    # Special treatment: Weigh values of df based on installed capacities in REMIND
    generation_shares = weigh_by_REMIND_capacity(generation_shares)

    # Throw warning if sum is not between 0.9999 and 1.0001 for each year and region (allowing for some numerical tolerance)
    if any(
        generation_shares.groupby(["year", "region"])["value"]
        .sum()
        .where(lambda x: (x < 0.99) | (x > 1.01))
        .notna()
    ):
        logger.warning(
            "Sum of generation shares is not between 0.99 and 1.01 for each year and region."
        )

    # %%
    # Export as csv values (informative purposes only, coupling parameters below via GDX)
    for fn, df in {
        "capacity_factors": capacity_factors,
        "availability_factors": availability_factors,
        "curtailments": curtailments,
        "generation_shares": generation_shares,
        "peak_residual_loads": peak_residual_loads,
        "grid_capacities": grid_capacities,
        "grid_investments": grid_investments,
        "preinstalled_capacities": preinstalled_capacities,
        "market_values": market_values,
        "hourly_prices": hourly_prices,
        "load_prices": load_prices,
        "electricity_prices_electrolysis": electricity_prices_electrolysis,
        "electricity_loads": electricity_loads,
        "generations": generations,
        "potentials": potentials,
        "optimal_capacities": optimal_capacities,
        "crossborder_flows": crossborder_flows,
        "crossborder_prices": crossborder_prices,
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
    s_epe_carrier = gt.Set(
        gdx,
        "electrolysis",
        records=sets["electrolysis"],
        description="Electricity price for electrolysis exported from PyPSAEur.",
    )

    # Now we can add data to the container
    c = gt.Parameter(
        gdx,
        name="capacity_factor",
        domain=[s_year, s_region, s_carrier],
        records=capacity_factors,
        description="Cacacity factors of technology per year and region in p.u.",
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

    xbf = gt.Parameter(
        gdx,
        name="crossborder_flow",
        domain=[s_year, s_region, s_region],
        records=crossborder_flows,
        description="Crossborder flows (exports) from region (columnn 2) to region (column 3) per year in MWh",
    )

    xbp = gt.Parameter(
        gdx,
        name="crossborder_price",
        domain=[s_year, s_region, s_region],
        records=crossborder_prices,
        description="Crossborder prices paid by the importing region (column 3) to the exporting region (column 2) per year in EUR/MWh",
    )

    grs = gt.Parameter(
        gdx,
        name="generation_region_share",
        domain=[s_year, s_region],
        records=generation_region_shares,
        description="Share of generation of region in total generation per year in p.u.",
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
        domain=[s_year, s_region, s_epe_carrier],
        records=electricity_prices_electrolysis,
        description="Electricity price for electrolysis per year and region in [EUR/MWh electricity] (weighted, based on the electricity drawn to meet the H2 demand from REMIND).",
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
