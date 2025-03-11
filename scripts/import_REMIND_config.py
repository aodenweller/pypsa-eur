# -*- coding: utf-8 -*-
# This file enables the scenario and iteration specific adjustment of the PyPSA config.yaml file
# Reading in the correct yaml file then requires use of the config_provider function in each rule
# The config_provider function has been adapted for that purpose
# While many changes could also be applied simply by reading in the REMIND2PyPSAEUR.gdx file in the
# corresponding places when creating the PyPSA network, this script also enables to change parameters that
# were defined previously in PyPSA. Therefore this script aims to minimise code changes in PyPSA.

#%%

import logging
import yaml
import copy

from _helpers import (
    configure_logging,
    mock_snakemake,
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "import_REMIND_config",
            scenario="TEST",
            iteration="11",
            year="2030"
        )

    configure_logging(snakemake)
    
#%%

    # Read standard config yaml file
    with open(snakemake.input["config_default"], "r", encoding="utf-8") as f:
        config_default = yaml.load(f, Loader=yaml.SafeLoader)
        
    # Make deep copy of config_default
    config = copy.deepcopy(config_default)
        
    # Example: Make sure [remind_coupling][links] contains "battery charger"
    #if "battery charger" not in config["remind_coupling"]["preinvestment_capacities"]["links"]:
    #    config["remind_coupling"]["preinvestment_capacities"]["links"].append("battery charger")

    # Write new config yaml file
    with open(snakemake.output["config"], "w") as f:
        yaml.dump(config,
                  f,
                  default_flow_style=False,
                  allow_unicode=True,
                  sort_keys=False,
                  encoding="utf-8")
