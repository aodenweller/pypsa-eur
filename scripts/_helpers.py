# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2017-2024 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT

import contextlib
import copy
import functools
import hashlib
import logging
import os
import re
import urllib
from functools import partial
from os.path import exists
from pathlib import Path
from shutil import copyfile

import pandas as pd
import pytz
import requests
import yaml
from snakemake.utils import update_config
from tqdm import tqdm

import subprocess
import sys
import time
import multiprocessing
import gurobipy as grb

logger = logging.getLogger(__name__)

REGION_COLS = ["geometry", "name", "x", "y", "country"]

DEFAULT_TUNNEL_PORT = 1080
LOGIN_NODE = "01"

def copy_default_files(workflow):
    default_files = {
        "config/config.default.yaml": "config/config.yaml",
        "config/scenarios.template.yaml": "config/scenarios.yaml",
    }
    for template, target in default_files.items():
        target = os.path.join(workflow.current_basedir, target)
        template = os.path.join(workflow.current_basedir, template)
        if not exists(target) and exists(template):
            copyfile(template, target)


def get_scenarios(run):
    scenario_config = run.get("scenarios", {})
    if run["name"] and scenario_config.get("enable"):
        fn = Path(scenario_config["file"])
        if fn.exists():
            scenarios = yaml.safe_load(fn.read_text())
            if run["name"] == "all":
                run["name"] = list(scenarios.keys())
            return scenarios
    return {}


def get_rdir(run):
    scenario_config = run.get("scenarios", {})
    if run["name"] and scenario_config.get("enable"):
        RDIR = "{run}/"
    elif run["name"]:
        RDIR = run["name"] + "/"
    else:
        RDIR = ""

    prefix = run.get("prefix", "")
    if prefix:
        RDIR = f"{prefix}/{RDIR}"

    return RDIR


def get_run_path(fn, dir, rdir, shared_resources, exclude_from_shared):
    """
    Dynamically provide paths based on shared resources and filename.

    Use this function for snakemake rule inputs or outputs that should be
    optionally shared across runs or created individually for each run.

    Parameters
    ----------
    fn : str
        The filename for the path to be generated.
    dir : str
        The base directory.
    rdir : str
        Relative directory for non-shared resources.
    shared_resources : str or bool
        Specifies which resources should be shared.
        - If string is "base", special handling for shared "base" resources (see notes).
        - If random string other than "base", this folder is used instead of the `rdir` keyword.
        - If boolean, directly specifies if the resource is shared.
    exclude_from_shared: list
        List of filenames to exclude from shared resources. Only relevant if shared_resources is "base".

    Returns
    -------
    str
        Full path where the resource should be stored.

    Notes
    -----
    Special case for "base" allows no wildcards other than "technology", "year"
    and "scope" and excludes filenames starting with "networks/elec" or
    "add_electricity". All other resources are shared.
    """
    if shared_resources == "base":
        pattern = r"\{([^{}]+)\}"
        existing_wildcards = set(re.findall(pattern, fn))
        irrelevant_wildcards = {"technology", "year", "scope", "kind"}
        no_relevant_wildcards = not existing_wildcards - irrelevant_wildcards
        not_shared_rule = (
            not fn.startswith("networks/elec")
            and not fn.startswith("add_electricity")
            and not any(fn.startswith(ex) for ex in exclude_from_shared)
        )
        is_shared = no_relevant_wildcards and not_shared_rule
        rdir = "" if is_shared else rdir
    elif isinstance(shared_resources, str):
        rdir = shared_resources + "/"
    elif isinstance(shared_resources, bool):
        rdir = "" if shared_resources else rdir
    else:
        raise ValueError(
            "shared_resources must be a boolean, str, or 'base' for special handling."
        )

    return f"{dir}{rdir}{fn}"


def path_provider(dir, rdir, shared_resources, exclude_from_shared):
    """
    Returns a partial function that dynamically provides paths based on shared
    resources and the filename.

    Returns
    -------
    partial function
        A partial function that takes a filename as input and
        returns the path to the file based on the shared_resources parameter.
    """
    return partial(
        get_run_path,
        dir=dir,
        rdir=rdir,
        shared_resources=shared_resources,
        exclude_from_shared=exclude_from_shared,
    )


def get_opt(opts, expr, flags=None):
    """
    Return the first option matching the regular expression.

    The regular expression is case-insensitive by default.
    """
    if flags is None:
        flags = re.IGNORECASE
    for o in opts:
        match = re.match(expr, o, flags=flags)
        if match:
            return match.group(0)
    return None


def find_opt(opts, expr):
    """
    Return if available the float after the expression.
    """
    for o in opts:
        if expr in o:
            m = re.findall(r"m?\d+(?:[\.p]\d+)?", o)
            if len(m) > 0:
                return True, float(m[-1].replace("p", ".").replace("m", "-"))
            else:
                return True, None
    return False, None


# Define a context manager to temporarily mute print statements
@contextlib.contextmanager
def mute_print():
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull):
            yield


def set_scenario_config(snakemake):
    scenario = snakemake.config["run"].get("scenarios", {})
    if scenario.get("enable") and "run" in snakemake.wildcards.keys():
        try:
            with open(scenario["file"], "r") as f:
                scenario_config = yaml.safe_load(f)
        except FileNotFoundError:
            # fallback for mock_snakemake
            script_dir = Path(__file__).parent.resolve()
            root_dir = script_dir.parent
            with open(root_dir / scenario["file"], "r") as f:
                scenario_config = yaml.safe_load(f)
        update_config(snakemake.config, scenario_config[snakemake.wildcards.run])


def configure_logging(snakemake, skip_handlers=False):
    """
    Configure the basic behaviour for the logging module.

    Note: Must only be called once from the __main__ section of a script.

    The setup includes printing log messages to STDERR and to a log file defined
    by either (in priority order): snakemake.log.python, snakemake.log[0] or "logs/{rulename}.log".
    Additional keywords from logging.basicConfig are accepted via the snakemake configuration
    file under snakemake.config.logging.

    Parameters
    ----------
    snakemake : snakemake object
        Your snakemake object containing a snakemake.config and snakemake.log.
    skip_handlers : True | False (default)
        Do (not) skip the default handlers created for redirecting output to STDERR and file.
    """
    import logging
    import sys

    kwargs = snakemake.config.get("logging", dict()).copy()
    kwargs.setdefault("level", "INFO")

    if skip_handlers is False:
        fallback_path = Path(__file__).parent.joinpath(
            "..", "logs", f"{snakemake.rule}.log"
        )
        logfile = snakemake.log.get(
            "python", snakemake.log[0] if snakemake.log else fallback_path
        )
        kwargs.update(
            {
                "handlers": [
                    # Prefer the 'python' log, otherwise take the first log for each
                    # Snakemake rule
                    logging.FileHandler(logfile),
                    logging.StreamHandler(),
                ]
            }
        )
    logging.basicConfig(**kwargs)

    # Setup a function to handle uncaught exceptions and include them with their stacktrace into logfiles
    def handle_exception(exc_type, exc_value, exc_traceback):
        # Log the exception
        logger = logging.getLogger()
        logger.error(
            "Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback)
        )

    sys.excepthook = handle_exception


def update_p_nom_max(n):
    # if extendable carriers (solar/onwind/...) have capacity >= 0,
    # e.g. existing assets from the OPSD project are included to the network,
    # the installed capacity might exceed the expansion limit.
    # Hence, we update the assumptions.

    n.generators.p_nom_max = n.generators[["p_nom_min", "p_nom_max"]].max(1)


def aggregate_p_nom(n):
    return pd.concat(
        [
            n.generators.groupby("carrier").p_nom_opt.sum(),
            n.storage_units.groupby("carrier").p_nom_opt.sum(),
            n.links.groupby("carrier").p_nom_opt.sum(),
            n.loads_t.p.groupby(n.loads.carrier, axis=1).sum().mean(),
        ]
    )


def aggregate_p(n):
    return pd.concat(
        [
            n.generators_t.p.sum().groupby(n.generators.carrier).sum(),
            n.storage_units_t.p.sum().groupby(n.storage_units.carrier).sum(),
            n.stores_t.p.sum().groupby(n.stores.carrier).sum(),
            -n.loads_t.p.sum().groupby(n.loads.carrier).sum(),
        ]
    )


def get(item, investment_year=None):
    """
    Check whether item depends on investment year.
    """
    if not isinstance(item, dict):
        return item
    elif investment_year in item.keys():
        return item[investment_year]
    else:
        logger.warning(
            f"Investment key {investment_year} not found in dictionary {item}."
        )
        keys = sorted(item.keys())
        if investment_year < keys[0]:
            logger.warning(f"Lower than minimum key. Taking minimum key {keys[0]}")
            return item[keys[0]]
        elif investment_year > keys[-1]:
            logger.warning(f"Higher than maximum key. Taking maximum key {keys[0]}")
            return item[keys[-1]]
        else:
            logger.warning(
                "Interpolate linearly between the next lower and next higher year."
            )
            lower_key = max(k for k in keys if k < investment_year)
            higher_key = min(k for k in keys if k > investment_year)
            lower = item[lower_key]
            higher = item[higher_key]
            return lower + (higher - lower) * (investment_year - lower_key) / (
                higher_key - lower_key
            )


def aggregate_e_nom(n):
    return pd.concat(
        [
            (n.storage_units["p_nom_opt"] * n.storage_units["max_hours"])
            .groupby(n.storage_units["carrier"])
            .sum(),
            n.stores["e_nom_opt"].groupby(n.stores.carrier).sum(),
        ]
    )


def aggregate_p_curtailed(n):
    return pd.concat(
        [
            (
                (
                    n.generators_t.p_max_pu.sum().multiply(n.generators.p_nom_opt)
                    - n.generators_t.p.sum()
                )
                .groupby(n.generators.carrier)
                .sum()
            ),
            (
                (n.storage_units_t.inflow.sum() - n.storage_units_t.p.sum())
                .groupby(n.storage_units.carrier)
                .sum()
            ),
        ]
    )


def aggregate_costs(n, flatten=False, opts=None, existing_only=False):
    components = dict(
        Link=("p_nom", "p0"),
        Generator=("p_nom", "p"),
        StorageUnit=("p_nom", "p"),
        Store=("e_nom", "p"),
        Line=("s_nom", None),
        Transformer=("s_nom", None),
    )

    costs = {}
    for c, (p_nom, p_attr) in zip(
        n.iterate_components(components.keys(), skip_empty=False), components.values()
    ):
        if c.df.empty:
            continue
        if not existing_only:
            p_nom += "_opt"
        costs[(c.list_name, "capital")] = (
            (c.df[p_nom] * c.df.capital_cost).groupby(c.df.carrier).sum()
        )
        if p_attr is not None:
            p = c.pnl[p_attr].sum()
            if c.name == "StorageUnit":
                p = p.loc[p > 0]
            costs[(c.list_name, "marginal")] = (
                (p * c.df.marginal_cost).groupby(c.df.carrier).sum()
            )
    costs = pd.concat(costs)

    if flatten:
        assert opts is not None
        conv_techs = opts["conv_techs"]

        costs = costs.reset_index(level=0, drop=True)
        costs = costs["capital"].add(
            costs["marginal"].rename({t: t + " marginal" for t in conv_techs}),
            fill_value=0.0,
        )

    return costs


def progress_retrieve(url, file, disable=False):
    if disable:
        urllib.request.urlretrieve(url, file)
    else:
        with tqdm(unit="B", unit_scale=True, unit_divisor=1024, miniters=1) as t:

            def update_to(b=1, bsize=1, tsize=None):
                if tsize is not None:
                    t.total = tsize
                t.update(b * bsize - t.n)

            urllib.request.urlretrieve(url, file, reporthook=update_to)


def mock_snakemake(
    rulename,
    root_dir=None,
    configfiles=None,
    submodule_dir="workflow/submodules/pypsa-eur",
    **wildcards,
):
    """
    This function is expected to be executed from the 'scripts'-directory of '
    the snakemake project. It returns a snakemake.script.Snakemake object,
    based on the Snakefile.

    If a rule has wildcards, you have to specify them in **wildcards.

    Parameters
    ----------
    rulename: str
        name of the rule for which the snakemake object should be generated
    root_dir: str/path-like
        path to the root directory of the snakemake project
    configfiles: list, str
        list of configfiles to be used to update the config
    submodule_dir: str, Path
        in case PyPSA-Eur is used as a submodule, submodule_dir is
        the path of pypsa-eur relative to the project directory.
    **wildcards:
        keyword arguments fixing the wildcards. Only necessary if wildcards are
        needed.
    """
    import os

    import snakemake as sm
    from pypsa.definitions.structures import Dict
    from snakemake.api import Workflow
    from snakemake.common import SNAKEFILE_CHOICES
    from snakemake.script import Snakemake
    from snakemake.settings.types import (
        ConfigSettings,
        DAGSettings,
        ResourceSettings,
        StorageSettings,
        WorkflowSettings,
    )

    script_dir = Path(__file__).parent.resolve()
    if root_dir is None:
        root_dir = script_dir.parent
    else:
        root_dir = Path(root_dir).resolve()

    user_in_script_dir = Path.cwd().resolve() == script_dir
    if str(submodule_dir) in __file__:
        # the submodule_dir path is only need to locate the project dir
        os.chdir(Path(__file__[: __file__.find(str(submodule_dir))]))
    elif user_in_script_dir:
        os.chdir(root_dir)
    elif Path.cwd().resolve() != root_dir:
        raise RuntimeError(
            "mock_snakemake has to be run from the repository root"
            f" {root_dir} or scripts directory {script_dir}"
        )
    try:
        # for p in SNAKEFILE_CHOICES:
        for p in ["Snakefile_REMIND"]:
            if os.path.exists(p):
                snakefile = p
                break
        if configfiles is None:
            configfiles = []
        elif isinstance(configfiles, str):
            configfiles = [configfiles]

        resource_settings = ResourceSettings()
        config_settings = ConfigSettings(configfiles=map(Path, configfiles))
        workflow_settings = WorkflowSettings()
        storage_settings = StorageSettings()
        dag_settings = DAGSettings(rerun_triggers=[])
        workflow = Workflow(
            config_settings,
            resource_settings,
            workflow_settings,
            storage_settings,
            dag_settings,
            storage_provider_settings=dict(),
        )
        workflow.include(snakefile)

        if configfiles:
            for f in configfiles:
                if not os.path.exists(f):
                    raise FileNotFoundError(f"Config file {f} does not exist.")
                workflow.configfile(f)

        workflow.global_resources = {}
        rule = workflow.get_rule(rulename)
        dag = sm.dag.DAG(workflow, rules=[rule])
        wc = Dict(wildcards)
        job = sm.jobs.Job(rule, dag, wc)

        def make_accessable(*ios):
            for io in ios:
                for i, _ in enumerate(io):
                    io[i] = os.path.abspath(io[i])

        make_accessable(job.input, job.output, job.log)
        snakemake = Snakemake(
            job.input,
            job.output,
            job.params,
            job.wildcards,
            job.threads,
            job.resources,
            job.log,
            job.dag.workflow.config,
            job.rule.name,
            None,
        )
        # create log and output dir if not existent
        for path in list(snakemake.log) + list(snakemake.output):
            Path(path).parent.mkdir(parents=True, exist_ok=True)

    finally:
        if user_in_script_dir:
            os.chdir(script_dir)
    return snakemake


def generate_periodic_profiles(dt_index, nodes, weekly_profile, localize=None):
    """
    Give a 24*7 long list of weekly hourly profiles, generate this for each
    country for the period dt_index, taking account of time zones and summer
    time.
    """
    weekly_profile = pd.Series(weekly_profile, range(24 * 7))

    week_df = pd.DataFrame(index=dt_index, columns=nodes)

    for node in nodes:
        ct = node[:2] if node[:2] != "XK" else "RS"
        timezone = pytz.timezone(pytz.country_timezones[ct][0])
        tz_dt_index = dt_index.tz_convert(timezone)
        week_df[node] = [24 * dt.weekday() + dt.hour for dt in tz_dt_index]
        week_df[node] = week_df[node].map(weekly_profile)

    week_df = week_df.tz_localize(localize)

    return week_df


def parse(infix):
    """
    Recursively parse a chained wildcard expression into a dictionary or a YAML
    object.

    Parameters
    ----------
    list_to_parse : list
        The list to parse.

    Returns
    -------
    dict or YAML object
        The parsed list.
    """
    if len(infix) == 1:
        return yaml.safe_load(infix[0])
    else:
        return {infix.pop(0): parse(infix)}


def update_config_from_wildcards(config, w, inplace=True):
    """
    Parses configuration settings from wildcards and updates the config.
    """

    if not inplace:
        config = copy.deepcopy(config)

    if w.get("opts"):
        opts = w.opts.split("-")

        if nhours := get_opt(opts, r"^\d+(h|seg)$"):
            config["clustering"]["temporal"]["resolution_elec"] = nhours

        co2l_enable, co2l_value = find_opt(opts, "Co2L")
        if co2l_enable:
            config["electricity"]["co2limit_enable"] = True
            if co2l_value is not None:
                config["electricity"]["co2limit"] = (
                    co2l_value * config["electricity"]["co2base"]
                )

        gasl_enable, gasl_value = find_opt(opts, "CH4L")
        if gasl_enable:
            config["electricity"]["gaslimit_enable"] = True
            if gasl_value is not None:
                config["electricity"]["gaslimit"] = gasl_value * 1e6

        if "Ept" in opts:
            config["costs"]["emission_prices"]["co2_monthly_prices"] = True

        ep_enable, ep_value = find_opt(opts, "Ep")
        if ep_enable:
            config["costs"]["emission_prices"]["enable"] = True
            if ep_value is not None:
                config["costs"]["emission_prices"]["co2"] = ep_value

        if "ATK" in opts:
            config["autarky"]["enable"] = True
            if "ATKc" in opts:
                config["autarky"]["by_country"] = True

        attr_lookup = {
            "p": "p_nom_max",
            "e": "e_nom_max",
            "c": "capital_cost",
            "m": "marginal_cost",
        }
        for o in opts:
            flags = ["+e", "+p", "+m", "+c"]
            if all(flag not in o for flag in flags):
                continue
            carrier, attr_factor = o.split("+")
            attr = attr_lookup[attr_factor[0]]
            factor = float(attr_factor[1:])
            if not isinstance(config["adjustments"]["electricity"], dict):
                config["adjustments"]["electricity"] = dict()
            update_config(
                config["adjustments"]["electricity"], {attr: {carrier: factor}}
            )

    if w.get("sector_opts"):
        opts = w.sector_opts.split("-")

        if "T" in opts:
            config["sector"]["transport"] = True

        if "H" in opts:
            config["sector"]["heating"] = True

        if "B" in opts:
            config["sector"]["biomass"] = True

        if "I" in opts:
            config["sector"]["industry"] = True

        if "A" in opts:
            config["sector"]["agriculture"] = True

        if "CCL" in opts:
            config["solving"]["constraints"]["CCL"] = True

        eq_value = get_opt(opts, r"^EQ+\d*\.?\d+(c|)")
        for o in opts:
            if eq_value is not None:
                config["solving"]["constraints"]["EQ"] = eq_value
            elif "EQ" in o:
                config["solving"]["constraints"]["EQ"] = True
            break

        if "BAU" in opts:
            config["solving"]["constraints"]["BAU"] = True

        if "SAFE" in opts:
            config["solving"]["constraints"]["SAFE"] = True

        if nhours := get_opt(opts, r"^\d+(h|sn|seg)$"):
            config["clustering"]["temporal"]["resolution_sector"] = nhours

        if "decentral" in opts:
            config["sector"]["electricity_transmission_grid"] = False

        if "noH2network" in opts:
            config["sector"]["H2_network"] = False

        if "nowasteheat" in opts:
            config["sector"]["use_fischer_tropsch_waste_heat"] = False
            config["sector"]["use_methanolisation_waste_heat"] = False
            config["sector"]["use_haber_bosch_waste_heat"] = False
            config["sector"]["use_methanation_waste_heat"] = False
            config["sector"]["use_fuel_cell_waste_heat"] = False
            config["sector"]["use_electrolysis_waste_heat"] = False

        if "nodistrict" in opts:
            config["sector"]["district_heating"]["progress"] = 0.0

        dg_enable, dg_factor = find_opt(opts, "dist")
        if dg_enable:
            config["sector"]["electricity_distribution_grid"] = True
            if dg_factor is not None:
                config["sector"][
                    "electricity_distribution_grid_cost_factor"
                ] = dg_factor

        if "biomasstransport" in opts:
            config["sector"]["biomass_transport"] = True

        _, maxext = find_opt(opts, "linemaxext")
        if maxext is not None:
            config["lines"]["max_extension"] = maxext * 1e3
            config["links"]["max_extension"] = maxext * 1e3

        _, co2l_value = find_opt(opts, "Co2L")
        if co2l_value is not None:
            config["co2_budget"] = float(co2l_value)

        if co2_distribution := get_opt(opts, r"^(cb)\d+(\.\d+)?(ex|be)$"):
            config["co2_budget"] = co2_distribution

        if co2_budget := get_opt(opts, r"^(cb)\d+(\.\d+)?$"):
            config["co2_budget"] = float(co2_budget[2:])

        attr_lookup = {
            "p": "p_nom_max",
            "e": "e_nom_max",
            "c": "capital_cost",
            "m": "marginal_cost",
        }
        for o in opts:
            flags = ["+e", "+p", "+m", "+c"]
            if all(flag not in o for flag in flags):
                continue
            carrier, attr_factor = o.split("+")
            attr = attr_lookup[attr_factor[0]]
            factor = float(attr_factor[1:])
            if not isinstance(config["adjustments"]["sector"], dict):
                config["adjustments"]["sector"] = dict()
            update_config(config["adjustments"]["sector"], {attr: {carrier: factor}})

        _, sdr_value = find_opt(opts, "sdr")
        if sdr_value is not None:
            config["costs"]["social_discountrate"] = sdr_value / 100

        _, seq_limit = find_opt(opts, "seq")
        if seq_limit is not None:
            config["sector"]["co2_sequestration_potential"] = seq_limit

        # any config option can be represented in wildcard
        for o in opts:
            if o.startswith("CF+"):
                infix = o.split("+")[1:]
                update_config(config, parse(infix))

    if not inplace:
        return config


@functools.lru_cache
def get_technology_mapping(
    fn: str or Path,
    group_technologies: bool = False,
):
    """
    Get a mapping between technologies in REMIND and PyPSA-EUR, inferred from
    the technology_cost_mapping file.

    Technologies can also be mapped to grouped technologies.

    Parameters
    ----------
    fn : str
        Path to the technology cost mapping file.
    group_technologies : bool, optional
        Whether to group technologies, by default False.

    Returns
    -------
    pd.DataFrame
        Dataframe containing the technology mapping with columns "PyPSA-Eur" and "REMIND-EU".
        If group_technologies is True, the dataframe will contain an additional column "technology_group".
    """

    new_mapping = pd.read_csv(fn)
    new_mapping = new_mapping.query("parameter == 'investment'")
    new_mapping = new_mapping.query(
        "`couple to` == 'mapping generation weighted to reference REMIND-EU technology'"
    )

    new_mapping = new_mapping[["PyPSA-EUR technology", "reference"]]
    # convert potential list-like entries to real list
    new_mapping["reference"] = new_mapping["reference"].map(yaml.safe_load)
    # Turn all list-entries into separate rows
    new_mapping = new_mapping.explode("reference")
    new_mapping = new_mapping.rename(
        columns={"PyPSA-EUR technology": "PyPSA-Eur", "reference": "REMIND-EU"}
    )

    if not (offwind := new_mapping.loc[new_mapping["PyPSA-Eur"] == "offwind"]).empty:
        logger.info(
            "'offwind' technology detected. Adding offwind-ac and offwind-dc to technology mapping..."
        )
        new_mapping = pd.concat(
            [
                new_mapping,
                offwind.replace("offwind", "offwind-ac"),
                offwind.replace("offwind", "offwind-dc"),
            ]
        ).reset_index(drop=True)

    if "hydro" in new_mapping["PyPSA-Eur"].unique() and (
        "ror" not in new_mapping["PyPSA-Eur"].unique()
        or "PHS" not in new_mapping["PyPSA-Eur"].unique()
    ):
        logger.info(
            "'hydro' technology but 'ror' and/or 'PHS' are missing. Adding 'ror' and 'PHS' to technology mapping..."
        )
        hydro = new_mapping.loc[new_mapping["PyPSA-Eur"] == "hydro"]
        new_mapping = pd.concat(
            [
                new_mapping,
                hydro.replace({"PyPSA-Eur": {"hydro": "ror"}}),
                hydro.replace({"PyPSA-Eur": {"hydro": "PHS"}}),
            ]
        ).reset_index(drop=True)

    # get all unique row combinations
    new_mapping = new_mapping.drop_duplicates().reset_index(drop=True)

    if group_technologies:
        # Determine PyPSA-Eur technologies/carriers which share the same constrained (= are mapped from the same REMIND technologies)
        new_mapping = (
            new_mapping.groupby("PyPSA-Eur")
            .agg(lambda x: tuple(sorted(x)))
            .reset_index()
            .groupby("REMIND-EU", as_index=False)
            .agg(lambda x: list(x))
        )

        # Create groups of PyPSA-Eur technologies, e.g. ['solar', 'solar rooftop'] -> "solar & solar rooftop"
        new_mapping["technology_group"] = new_mapping["PyPSA-Eur"].apply(
            lambda x: " & ".join(x)
        )
        new_mapping = new_mapping.explode("REMIND-EU").explode("PyPSA-Eur")

    return new_mapping


def get_region_mapping(
    fn,
    source: str = "REMIND-EU",
    target: str = "PyPSA-EUR",
    flatten: bool = False,
) -> dict:
    """
    Get a mapping between regions in REMIND and PyPSA-EUR.

    The mapping from REMIND-EU between regions and countries is read from file (fn),
    which is directly taken from the REMIND-EU model.
    The corresponding countries in PyPSA-EUR are determined using the country_converter.
    Valid values for source and target are: remind, pypsa-eur.
    Lower and uppercase are ignored.

    Parameters
    ----------
    fn : str
        Path to the region mapping file from REMIND-EU.
    source : str, optional
        Region mapping source, by default "remind-eu"
    target : str, optional
        Region mapping target, by default "pypsa-eur"
    flatten : bool, optional
        Whether to try to flatten the mapping; only valid
        if the mapping is unique for all keys.
        Default False.
    """
    import country_converter as coco

    # Create region mapping by loading the original mapping from REMIND-EU from file
    # and then mapping ISO 3166-1 alpha-3 country codes to PyPSA-EUR ISO 3166-1 alpha-2 country codes
    logger.info("Loading region mapping...")
    region_mapping = pd.read_csv(fn, sep=";").rename(
        columns={"RegionCode": "remind-eu"}
    )

    region_mapping["pypsa-eur"] = coco.convert(
        names=region_mapping["CountryCode"], to="ISO2"
    )
    region_mapping = region_mapping[["pypsa-eur", "remind-eu"]]

    # Append Kosovo to region mapping, not present in standard mapping and uses non-standard "KV" in PyPSA-EUR
    logger.info(
        "Manually adding Kosovo to region mapping (PyPSA-EUR: KV, REMIND-EU: part of NES region) ..."
    )
    region_mapping = pd.concat(
        [
            region_mapping,
            pd.DataFrame(
                {
                    "remind-eu": "NES",
                    "pypsa-eur": "KV",
                },
                columns=["pypsa-eur", "remind-eu"],
                index=[0],
            ),
        ]
    ).reset_index(drop=True)

    region_mapping = (
        region_mapping.groupby(source.lower())[target.lower()]
        .apply("unique")
        .apply(list)
    )

    if flatten:
        if (region_mapping.apply(lambda x: len(x)) != 1).any():
            logger.error(f"Cannot flatten mapping. Non-unique map contained:\n {df}")

        region_mapping = region_mapping.apply(lambda x: x[0])

    return region_mapping.to_dict()


def read_remind_data(file_path, variable_name, rename_columns={}, error_on_empty=True):
    """
    Auxiliary function for standardised and cached reading of REMIND-EU data
    files to pandas.DataFrame.

    Here all values read are considered variable, i.e. use
    "variable_name" also for what is considered a "parameter" in the GDX
    file.
    """
    from gamspy import Container

    @functools.lru_cache
    def _read_and_cache_remind_file(fp):
        return Container(load_from=fp)

    data = _read_and_cache_remind_file(file_path)[variable_name]
    df = data.records

    if error_on_empty and (df is None or df.empty):
        raise ValueError(f"{variable_name} is empty. In: {file_path}")

    df = df.rename(columns=rename_columns, errors="raise")

    return df


@functools.lru_cache
def get_technology_mapping(
    fn: str or Path,
    group_technologies: bool = False,
):
    """
    Get a mapping between technologies in REMIND and PyPSA-EUR, inferred from
    the technology_cost_mapping file.

    Technologies can also be mapped to grouped technologies.

    Parameters
    ----------
    fn : str
        Path to the technology cost mapping file.
    group_technologies : bool, optional
        Whether to group technologies, by default False.

    Returns
    -------
    pd.DataFrame
        Dataframe containing the technology mapping with columns "PyPSA-Eur" and "REMIND-EU".
        If group_technologies is True, the dataframe will contain an additional column "technology_group".
    """

    new_mapping = pd.read_csv(fn)
    new_mapping = new_mapping.query("parameter == 'investment'")
    new_mapping = new_mapping.query(
        "`couple to` == 'mapping generation weighted to reference REMIND-EU technology'"
    )

    new_mapping = new_mapping[["PyPSA-EUR technology", "reference"]]
    # convert potential list-like entries to real list
    new_mapping["reference"] = new_mapping["reference"].map(yaml.safe_load)
    # Turn all list-entries into separate rows
    new_mapping = new_mapping.explode("reference")
    new_mapping = new_mapping.rename(
        columns={"PyPSA-EUR technology": "PyPSA-Eur", "reference": "REMIND-EU"}
    )

    if not (offwind := new_mapping.loc[new_mapping["PyPSA-Eur"] == "offwind"]).empty:
        logger.info(
            "'offwind' technology detected. Adding offwind-ac and offwind-dc to technology mapping..."
        )
        new_mapping = pd.concat(
            [
                new_mapping,
                offwind.replace("offwind", "offwind-ac"),
                offwind.replace("offwind", "offwind-dc"),
            ]
        ).reset_index(drop=True)

    if "hydro" in new_mapping["PyPSA-Eur"].unique() and (
        "ror" not in new_mapping["PyPSA-Eur"].unique()
        or "PHS" not in new_mapping["PyPSA-Eur"].unique()
    ):
        logger.info(
            "'hydro' technology but 'ror' and/or 'PHS' are missing. Adding 'ror' and 'PHS' to technology mapping..."
        )
        hydro = new_mapping.loc[new_mapping["PyPSA-Eur"] == "hydro"]
        new_mapping = pd.concat(
            [
                new_mapping,
                hydro.replace({"PyPSA-Eur": {"hydro": "ror"}}),
                hydro.replace({"PyPSA-Eur": {"hydro": "PHS"}}),
            ]
        ).reset_index(drop=True)

    # get all unique row combinations
    new_mapping = new_mapping.drop_duplicates().reset_index(drop=True)

    if group_technologies:
        # Determine PyPSA-Eur technologies/carriers which share the same constrained (= are mapped from the same REMIND technologies)
        new_mapping = (
            new_mapping.groupby("PyPSA-Eur")
            .agg(lambda x: tuple(sorted(x)))
            .reset_index()
            .groupby("REMIND-EU", as_index=False)
            .agg(lambda x: list(x))
        )

        # Create groups of PyPSA-Eur technologies, e.g. ['solar', 'solar rooftop'] -> "solar & solar rooftop"
        new_mapping["technology_group"] = new_mapping["PyPSA-Eur"].apply(
            lambda x: " & ".join(x)
        )
        new_mapping = new_mapping.explode("REMIND-EU").explode("PyPSA-Eur")

    return new_mapping


def get_region_mapping(
    fn,
    source: str = "REMIND-EU",
    target: str = "PyPSA-EUR",
    flatten: bool = False,
) -> dict:
    """
    Get a mapping between regions in REMIND and PyPSA-EUR.

    The mapping from REMIND-EU between regions and countries is read from file (fn),
    which is directly taken from the REMIND-EU model.
    The corresponding countries in PyPSA-EUR are determined using the country_converter.
    Valid values for source and target are: remind, pypsa-eur.
    Lower and uppercase are ignored.

    Parameters
    ----------
    fn : str
        Path to the region mapping file from REMIND-EU.
    source : str, optional
        Region mapping source, by default "remind-eu"
    target : str, optional
        Region mapping target, by default "pypsa-eur"
    flatten : bool, optional
        Whether to try to flatten the mapping; only valid
        if the mapping is unique for all keys.
        Default False.
    """
    import country_converter as coco

    # Create region mapping by loading the original mapping from REMIND-EU from file
    # and then mapping ISO 3166-1 alpha-3 country codes to PyPSA-EUR ISO 3166-1 alpha-2 country codes
    logger.info("Loading region mapping...")
    region_mapping = pd.read_csv(fn, sep=";").rename(
        columns={"RegionCode": "remind-eu"}
    )

    region_mapping["pypsa-eur"] = coco.convert(
        names=region_mapping["CountryCode"], to="ISO2"
    )
    region_mapping = region_mapping[["pypsa-eur", "remind-eu"]]

    # Append Kosovo to region mapping, not present in standard mapping and uses non-standard "KV" in PyPSA-EUR
    logger.info(
        "Manually adding Kosovo to region mapping (PyPSA-EUR: KV, REMIND-EU: part of NES region) ..."
    )
    region_mapping = pd.concat(
        [
            region_mapping,
            pd.DataFrame(
                {
                    "remind-eu": "NES",
                    "pypsa-eur": "KV",
                },
                columns=["pypsa-eur", "remind-eu"],
                index=[0],
            ),
        ]
    ).reset_index(drop=True)

    region_mapping = (
        region_mapping.groupby(source.lower())[target.lower()]
        .apply("unique")
        .apply(list)
    )

    if flatten:
        if (region_mapping.apply(lambda x: len(x)) != 1).any():
            logger.error(f"Cannot flatten mapping. Non-unique map contained:\n {df}")

        region_mapping = region_mapping.apply(lambda x: x[0])

    return region_mapping.to_dict()


def read_remind_data(file_path, variable_name, rename_columns={}, error_on_empty=True):
    """
    Auxiliary function for standardised and cached reading of REMIND-EU data
    files to pandas.DataFrame.

    Here all values read are considered variable, i.e. use
    "variable_name" also for what is considered a "parameter" in the GDX
    file.
    """
    from gamspy import Container

    @functools.lru_cache
    def _read_and_cache_remind_file(fp):
        return Container(load_from=fp)

    data = _read_and_cache_remind_file(file_path)[variable_name]
    df = data.records

    if error_on_empty and (df is None or df.empty):
        raise ValueError(f"{variable_name} is empty. In: {file_path}")

    df = df.rename(columns=rename_columns, errors="raise")

    return df


@functools.lru_cache
def get_technology_mapping(
    fn: str or Path,
    group_technologies: bool = False,
):
    """
    Get a mapping between technologies in REMIND and PyPSA-EUR, inferred from
    the technology_cost_mapping file.

    Technologies can also be mapped to grouped technologies.

    Parameters
    ----------
    fn : str
        Path to the technology cost mapping file.
    group_technologies : bool, optional
        Whether to group technologies, by default False.

    Returns
    -------
    pd.DataFrame
        Dataframe containing the technology mapping with columns "PyPSA-Eur" and "REMIND-EU".
        If group_technologies is True, the dataframe will contain an additional column "technology_group".
    """

    new_mapping = pd.read_csv(fn)
    new_mapping = new_mapping.query("parameter == 'investment'")
    new_mapping = new_mapping.query(
        "`couple to` == 'mapping generation weighted to reference REMIND-EU technology'"
    )

    new_mapping = new_mapping[["PyPSA-EUR technology", "reference"]]
    # convert potential list-like entries to real list
    new_mapping["reference"] = new_mapping["reference"].map(yaml.safe_load)
    # Turn all list-entries into separate rows
    new_mapping = new_mapping.explode("reference")
    new_mapping = new_mapping.rename(
        columns={"PyPSA-EUR technology": "PyPSA-Eur", "reference": "REMIND-EU"}
    )

    if not (offwind := new_mapping.loc[new_mapping["PyPSA-Eur"] == "offwind"]).empty:
        logger.info(
            "'offwind' technology detected. Adding offwind-ac and offwind-dc to technology mapping..."
        )
        new_mapping = pd.concat(
            [
                new_mapping,
                offwind.replace("offwind", "offwind-ac"),
                offwind.replace("offwind", "offwind-dc"),
            ]
        ).reset_index(drop=True)

    if "hydro" in new_mapping["PyPSA-Eur"].unique() and (
        "ror" not in new_mapping["PyPSA-Eur"].unique()
        or "PHS" not in new_mapping["PyPSA-Eur"].unique()
    ):
        logger.info(
            "'hydro' technology but 'ror' and/or 'PHS' are missing. Adding 'ror' and 'PHS' to technology mapping..."
        )
        hydro = new_mapping.loc[new_mapping["PyPSA-Eur"] == "hydro"]
        new_mapping = pd.concat(
            [
                new_mapping,
                hydro.replace({"PyPSA-Eur": {"hydro": "ror"}}),
                hydro.replace({"PyPSA-Eur": {"hydro": "PHS"}}),
            ]
        ).reset_index(drop=True)

    # get all unique row combinations
    new_mapping = new_mapping.drop_duplicates().reset_index(drop=True)

    if group_technologies:
        # Determine PyPSA-Eur technologies/carriers which share the same constrained (= are mapped from the same REMIND technologies)
        new_mapping = (
            new_mapping.groupby("PyPSA-Eur")
            .agg(lambda x: tuple(sorted(x)))
            .reset_index()
            .groupby("REMIND-EU", as_index=False)
            .agg(lambda x: list(x))
        )

        # Create groups of PyPSA-Eur technologies, e.g. ['solar', 'solar rooftop'] -> "solar & solar rooftop"
        new_mapping["technology_group"] = new_mapping["PyPSA-Eur"].apply(
            lambda x: " & ".join(x)
        )
        new_mapping = new_mapping.explode("REMIND-EU").explode("PyPSA-Eur")

    return new_mapping


def get_region_mapping(
    fn,
    source: str = "REMIND-EU",
    target: str = "PyPSA-EUR",
    flatten: bool = False,
) -> dict:
    """
    Get a mapping between regions in REMIND and PyPSA-EUR.

    The mapping from REMIND-EU between regions and countries is read from file (fn),
    which is directly taken from the REMIND-EU model.
    The corresponding countries in PyPSA-EUR are determined using the country_converter.
    Valid values for source and target are: remind, pypsa-eur.
    Lower and uppercase are ignored.

    Parameters
    ----------
    fn : str
        Path to the region mapping file from REMIND-EU.
    source : str, optional
        Region mapping source, by default "remind-eu"
    target : str, optional
        Region mapping target, by default "pypsa-eur"
    flatten : bool, optional
        Whether to try to flatten the mapping; only valid
        if the mapping is unique for all keys.
        Default False.
    """
    import country_converter as coco

    # Create region mapping by loading the original mapping from REMIND-EU from file
    # and then mapping ISO 3166-1 alpha-3 country codes to PyPSA-EUR ISO 3166-1 alpha-2 country codes
    logger.info("Loading region mapping...")
    region_mapping = pd.read_csv(fn, sep=";").rename(
        columns={"RegionCode": "remind-eu"}
    )

    region_mapping["pypsa-eur"] = coco.convert(
        names=region_mapping["CountryCode"], to="ISO2"
    )
    region_mapping = region_mapping[["pypsa-eur", "remind-eu"]]

    # Append Kosovo to region mapping, not present in standard mapping and uses non-standard "KV" in PyPSA-EUR
    logger.info(
        "Manually adding Kosovo to region mapping (PyPSA-EUR: KV, REMIND-EU: part of NES region) ..."
    )
    region_mapping = pd.concat(
        [
            region_mapping,
            pd.DataFrame(
                {
                    "remind-eu": "NES",
                    "pypsa-eur": "KV",
                },
                columns=["pypsa-eur", "remind-eu"],
                index=[0],
            ),
        ]
    ).reset_index(drop=True)

    region_mapping = (
        region_mapping.groupby(source.lower())[target.lower()]
        .apply("unique")
        .apply(list)
    )

    if flatten:
        if (region_mapping.apply(lambda x: len(x)) != 1).any():
            logger.error(f"Cannot flatten mapping. Non-unique map contained:\n {df}")

        region_mapping = region_mapping.apply(lambda x: x[0])

    return region_mapping.to_dict()


def read_remind_data(file_path, variable_name, rename_columns={}, error_on_empty=True):
    """
    Auxiliary function for standardised and cached reading of REMIND-EU data
    files to pandas.DataFrame.

    Here all values read are considered variable, i.e. use
    "variable_name" also for what is considered a "parameter" in the GDX
    file.
    """
    from gamspy import Container

    @functools.lru_cache
    def _read_and_cache_remind_file(fp):
        return Container(load_from=fp)

    data = _read_and_cache_remind_file(file_path)[variable_name]
    df = data.records

    if error_on_empty and (df is None or df.empty):
        raise ValueError(f"{variable_name} is empty. In: {file_path}")

    df = df.rename(columns=rename_columns, errors="raise")

    return df


def get_checksum_from_zenodo(file_url):
    parts = file_url.split("/")
    record_id = parts[parts.index("records") + 1]
    filename = parts[-1]

    response = requests.get(f"https://zenodo.org/api/records/{record_id}", timeout=30)
    response.raise_for_status()
    data = response.json()

    for file in data["files"]:
        if file["key"] == filename:
            return file["checksum"]
    return None


def validate_checksum(file_path, zenodo_url=None, checksum=None):
    """
    Validate file checksum against provided or Zenodo-retrieved checksum.
    Calculates the hash of a file using 64KB chunks. Compares it against a
    given checksum or one from a Zenodo URL.

    Parameters
    ----------
    file_path : str
        Path to the file for checksum validation.
    zenodo_url : str, optional
        URL of the file on Zenodo to fetch the checksum.
    checksum : str, optional
        Checksum (format 'hash_type:checksum_value') for validation.

    Raises
    ------
    AssertionError
        If the checksum does not match, or if neither `checksum` nor `zenodo_url` is provided.


    Examples
    --------
    >>> validate_checksum("/path/to/file", checksum="md5:abc123...")
    >>> validate_checksum(
    ...     "/path/to/file",
    ...     zenodo_url="https://zenodo.org/records/12345/files/example.txt",
    ... )

    If the checksum is invalid, an AssertionError will be raised.
    """
    assert checksum or zenodo_url, "Either checksum or zenodo_url must be provided"
    if zenodo_url:
        checksum = get_checksum_from_zenodo(zenodo_url)
    hash_type, checksum = checksum.split(":")
    hasher = hashlib.new(hash_type)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):  # 64kb chunks
            hasher.update(chunk)
    calculated_checksum = hasher.hexdigest()
    assert (
        calculated_checksum == checksum
    ), "Checksum is invalid. This may be due to an incomplete download. Delete the file and re-execute the rule."


def get_snapshots(snapshots, drop_leap_day=False, freq="h", **kwargs):
    """
    Returns pandas DateTimeIndex potentially without leap days.
    """

    time = pd.date_range(freq=freq, **snapshots, **kwargs)
    if drop_leap_day and time.is_leap_year.any():
        time = time[~((time.month == 2) & (time.day == 29))]

    return time


def is_tunnel_alive(tunnel_config: dict):
    """Check if the SSH tunnel is running by checking if the port is in use."""
    port = tunnel_config.get("tunnel_port", DEFAULT_TUNNEL_PORT)
    result = subprocess.run(
        f"netstat -tlnp 2>/dev/null | grep :{port}", 
        shell=True, 
        stdout=subprocess.PIPE
    )
    return result.returncode == 0

def setup_gurobi_tunnel_and_env(
    tunnel_config: dict, logger: logging.Logger = None, attempts=4
) -> subprocess.Popen:
    """A utility function to set up the Gurobi environment variables and establish an
    SSH tunnel on HPCs. Otherwise the license check will fail if the compute nodes do
     not have internet access or a token server isn't set up

    Args:
        config (dict): the snakemake pypsa-china configuration
        logger (logging.Logger, optional): Logger. Defaults to None.
        attempts (int, optional): ssh connection attemps. Defaults to 4.
    """
    if not tunnel_config.get("use_tunnel", False):
        return
    logger.info("Setting up tunnel")
    user = os.getenv("USER")  # User is pulled from the environment
    port = tunnel_config.get("tunnel_port", DEFAULT_TUNNEL_PORT)

    # bash commands for tunnel: reduce pipe err severity (too high from snakemake)
    pipe_err = "set -o pipefail; "
    # ssh_command = f"ssh -vvv -fN -D {port} {user}@login{LOGIN_NODE}"
    ssh_command = f"ssh -o ServerAliveInterval=20 -o ServerAliveCountMax=10 -vvv -fN -D {port} {user}@login{LOGIN_NODE}"
    logger.info(f"Attempting ssh tunnel to login node {LOGIN_NODE}")
    # Kill any existing SSH tunnel on that port before starting a new one
    subprocess.run(f"pkill -f '{ssh_command}'", shell=True)
    # Run SSH in the background to establish the tunnel
    socks_proc = subprocess.Popen(
        pipe_err + ssh_command, shell=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE
    )
    try:
        time.sleep(1)
        # [-1] because ssh is last command
        err = socks_proc.communicate(timeout=5)[-1].decode()
        logger.info(f"ssh err returns {str(err)}")
        if err.find("Permission") != -1 or err.find("Could not resolve hostname") != -1:
            socks_proc.kill()
            time.sleep(2)  # Small delay before retrying
        else:
            logger.info("Gurobi Environment variables & tunnel set up successfully.")
    except subprocess.TimeoutExpired:
        logger.info(
            f"SSH tunnel established on port {port} with possible errors (err check timedout)."
        )

    os.environ["https_proxy"] = f"socks5://127.0.0.1:{port}"
    os.environ["SSL_CERT_FILE"] = "/p/projects/rd3mod/ssl/ca-bundle.pem_2022-02-08"
    os.environ["GRB_CAFILE"] = "/p/projects/rd3mod/ssl/ca-bundle.pem_2022-02-08"

    # Set up Gurobi environment variables
    os.environ["GUROBI_HOME"] = "/p/projects/rd3mod/gurobi1103/linux64"
    os.environ["PATH"] += f":{os.environ['GUROBI_HOME']}/bin"
    os.environ["LD_LIBRARY_PATH"] += f":{os.environ['GUROBI_HOME']}/lib"
    os.environ["GRB_LICENSE_FILE"] = "/p/projects/rd3mod/gurobi_rc/gurobi.lic"
    os.environ["GRB_CURLVERBOSE"] = "1"
    os.environ["GRB_SERVER_TIMEOUT"] = "10"

    return socks_proc

def _check_gurobi_license_subprocess():
    """
    Subprocess function to check Gurobi license availability.
    This function will start the Gurobi environment to verify if a license is available.
    """
    try:
        env = grb.Env(empty=True)
        env.start()  # Start the Gurobi environment (this will attempt to acquire the license)
        logger.info("Gurobi license is available.")
        env.dispose()  # Dispose of the environment after use
        return True
    except grb.GurobiError as e:
        logger.error(f"Error checking Gurobi license: {e}")
        return False

def check_gurobi_license(attempts=5, timeout=10):
    """
    Checks the availability of the Gurobi license in a subprocess with timeout.
    Retries a few times if the license is not available.
    
    Parameters:
    - attempts: Number of attempts.
    - timeout: Time to wait before retrying (in seconds).
    
    Returns:
    - True if the license is available, False if the check times out.
    """
    logger.info("Checking Gurobi license availability...")
    
    for attempt in range(1, attempts + 1):
        # Create a multiprocessing Process to check license
        process = multiprocessing.Process(target=_check_gurobi_license_subprocess)
        process.start()
        
        process.join(timeout=timeout)  # Wait for the process to finish or timeout
        
        if process.is_alive():
            # If the process is still alive after the timeout, terminate it
            process.terminate()
            process.join()  # Ensure it is properly joined to clean up
            logger.warning(f"License check timeout. Retrying...")
        else:
            # If the process completed, check the result
            if process.exitcode == 0:
                # License was available
                return True
            else:
                # License was not available
                logger.warning("License not available during subprocess check. Retrying...")

    raise(RuntimeError("Gurobi license check failed after all attempts, aborting."))