# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2020-2024 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
"""
Retrieve powerplants from powerplantmatching (for REMIND coupling).
"""

import logging
import powerplantmatching as pm

from _helpers import (
    configure_logging,
    set_scenario_config,
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("retrieve_powerplants")
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    powerplants = pm.powerplants(from_url=True)
    
    powerplants.to_csv(snakemake.output[0]) 