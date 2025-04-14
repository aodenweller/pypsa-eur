# -*- coding: utf-8 -*-
# This script enables scenario and iteration specific changes to the PyPSA config.yaml file
# Reading in the correct yaml file then requires use of the config_provider function in each rule
# The config_provider function has been adapted for that purpose
# This script also creates a csv file for snakemake paramspace with the co2 price from REMIND
# This could also include the tempmoral resolution

# %%
import logging
import yaml
import copy
from functools import reduce
import numpy as np
import pandas as pd

from _helpers import (
    configure_logging,
    mock_snakemake,
    read_remind_data,
    get_region_mapping,
)

logger = logging.getLogger(__name__)


# Function to update nested config attributes dynamically
def set_nested_value(config, key_path, value):
    keys = key_path.split(".")
    last_key = keys.pop()
    nested_dict = reduce(lambda d, k: d.setdefault(k, {}), keys, config)
    nested_dict[last_key] = value


# Convert numpy types to native Python types
def convert_to_native_type(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()  # Converts NumPy scalar to Python native type
    return value  # Return value as is if no conversion is needed


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "import_REMIND_config",
            configfiles="config/config.remind.yaml",
            scenario="PyPSA_PkBudg1000_DEU_allRCL_PyPSArefactor_4nodes_loadsh_costRCL_2025-04-14_10.33.02",
            iteration="1",
        )

    configure_logging(snakemake)

    # %%
    ## First, read in the REMIND data and update the PyPSA config.yaml file

    # Read standard config yaml file
    with open(snakemake.input["config_default"], "r", encoding="utf-8") as f:
        config_default = yaml.load(f, Loader=yaml.SafeLoader)

    # Make deep copy of config_default
    config = copy.deepcopy(config_default)

    # Read in remind2config yaml file
    with open(snakemake.input["remind2config"], "r", encoding="utf-8") as f:
        remind2config = yaml.load(f, Loader=yaml.SafeLoader)

    # For each key in remind2config, update the corresponding key in config
    for _, df in remind2config.items():
        remind_switch = df["remind_switch"]
        overwrite_pypsa = df["overwrite_pypsa"]
        # Check if df has switch_mapping with df as a dictionary
        if "switch_mapping" in df:
            switch_mapping = df["switch_mapping"]
        else:
            switch_mapping = None
        # Read REMIND data
        try:
            remind_data = read_remind_data(
                snakemake.input["remind_config"], remind_switch
            )
        except KeyError:
            logging.warning(
                f"Switch {remind_switch} not found in REMIND data, skipping."
            )
            continue
        except IndexError:
            logging.warning(
                f"No values found for switch {remind_switch} in REMIND data, skipping."
            )
            continue
        except Exception as e:
            logging.warning(
                f"Error reading switch {remind_switch} from REMIND data: {e}, skipping."
            )
            continue
        switch_value = convert_to_native_type(remind_data["value"][0])
        # Get the mapping for the switch value
        new_value = switch_mapping.get(switch_value) if switch_mapping else switch_value
        # Update config
        set_nested_value(config, overwrite_pypsa, new_value)

    # Write new config yaml file
    with open(snakemake.output["config"], "w") as f:
        yaml.dump(
            config,
            f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
            encoding="utf-8",
        )

    # %%
    ## Second, read in REMIND data and create csv for snakemake paramspace

    # Load and transform region mapping
    region_mapping = get_region_mapping(
        snakemake.input["region_mapping"], source="PyPSA-EUR", target="REMIND-EU"
    )
    region_mapping = pd.DataFrame(region_mapping).T.reset_index()
    region_mapping.columns = ["PyPSA-EUR", "REMIND-EU"]
    region_mapping = region_mapping.loc[
        region_mapping["PyPSA-EUR"].isin(snakemake.config["countries"])
    ]

    df = read_remind_data(
        file_path=snakemake.input["remind_data"],
        variable_name="p_priceCO2",
        rename_columns={
            "tall": "year",
            "all_regi": "region",
        },
    )

    # unit conversion from USD/tC to USD/tCO2
    df["value"] *= 12 / (12 + 2 * 16)

    # Get coupled years from REMIND data
    years_coupled = (
        read_remind_data(
            file_path=snakemake.input["remind_data"],
            variable_name="tPy32",
            rename_columns={"ttot": "year"},
        )
        .year.unique()
        .tolist()
    )

    # Calculate mean co2 price across all regions overlapping between REMIND and PyPSA-EUR countries for each year
    # TODO: Implement regional prices in PyPSA!
    df = (
        df.loc[df["region"].isin(region_mapping["REMIND-EU"])]
        .groupby("year")["value"]
        .mean()
    )

    # add all years from variable as additional indices to df
    # ensure same dtypes for index are int, else df.reindex will produce wrong results
    df.index = df.index.astype(int)

    df = (
        df.reindex(
            list(map(int, years_coupled)),
            fill_value=0,
        )
        .to_frame("co2_price")
        .reset_index()
    )

    # Create a csv file which can be directly read by snakemake paramspace
    # need to add the remaining wildcards of interest, assume each wildcard has only one
    # entry in the config.yaml file
    if (
        len(snakemake.config["scenario"]["ll"]) != 1
        or len(snakemake.config["scenario"]["simpl"]) != 1
        or len(snakemake.config["scenario"]["clusters"]) != 1
        or len(snakemake.config["scenario"]["opts"]) != 1
    ):
        logger.error(
            "Only exactly one entry for config['scenario'] -> simpl, clusters, ll and opts in config permitted."
        )

    df["scenario"] = snakemake.wildcards["scenario"]
    df["iteration"] = snakemake.wildcards["iteration"]
    df["simpl"] = snakemake.config["scenario"]["simpl"][0]
    df["clusters"] = snakemake.config["scenario"]["clusters"][0]
    df["ll"] = snakemake.config["scenario"]["ll"][0]
    df["opts"] = snakemake.config["scenario"]["opts"][0]

    # Update number of clusters to value provided in REMIND switch
    try:
        n_clusters = read_remind_data(
            snakemake.input["remind_config"], "c32_pypsa_cfg_nodes"
        )["value"][0]
        df["clusters"] = int(n_clusters)
    except KeyError:
        logging.warning("Switch 'c32_pypsa_cfg_nodes' not found in REMIND data, skipping.")

    # Update temporal resolution to value provided in REMIND switch
    try:
        temporal_resolution = read_remind_data(
            snakemake.input["remind_config"], "c32_pypsa_cfg_hourly_res"
        )
        temporal_resolution = int(temporal_resolution.value.iloc[0])
        df["opts"] = df["opts"] = df["opts"].str.replace(
            r"\b\d{1,2}(?=H)", str(temporal_resolution), regex=True
        )
    except KeyError:
        logging.warning(
            "Switch 'c32_pypsa_cfg_hourly_res' not found in REMIND data, skipping."
        )

    # Preserve all opts and substitute the co2 price placeholder ("EpREMIND") with the actual co2 price from REMIND
    if not df["opts"].str.contains("-EpREMIND").all():
        logging.error("Placeholder '-EpREMIND' missing from config['scenario']['opts']")
    df["opts"] = df.apply(
        lambda row: row["opts"].replace("-EpREMIND", f"-Ep{row['co2_price']:0.1f}"),
        axis="columns",
    )
    df = df.drop(columns=["co2_price"])

    df.to_csv(snakemake.output["scenario_wildcards"], index=False)

# %%
