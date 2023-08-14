# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2017-2023 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT

import contextlib
import logging
import os
import urllib
from pathlib import Path

import pandas as pd
import pytz
import yaml
from pypsa.components import component_attrs, components
from pypsa.descriptors import Dict
from tqdm import tqdm

logger = logging.getLogger(__name__)

REGION_COLS = ["geometry", "name", "x", "y", "country"]


# Define a context manager to temporarily mute print statements
@contextlib.contextmanager
def mute_print():
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull):
            yield


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


def mock_snakemake(rulename, configfiles=[], **wildcards):
    """
    This function is expected to be executed from the 'scripts'-directory of '
    the snakemake project. It returns a snakemake.script.Snakemake object,
    based on the Snakefile.

    If a rule has wildcards, you have to specify them in **wildcards.

    Parameters
    ----------
    rulename: str
        name of the rule for which the snakemake object should be generated
    configfiles: list, str
        list of configfiles to be used to update the config
    **wildcards:
        keyword arguments fixing the wildcards. Only necessary if wildcards are
        needed.
    """
    import os

    import snakemake as sm
    from packaging.version import Version, parse
    from pypsa.descriptors import Dict
    from snakemake.script import Snakemake

    script_dir = Path(__file__).parent.resolve()
    root_dir = script_dir.parent

    user_in_script_dir = Path.cwd().resolve() == script_dir
    if user_in_script_dir:
        os.chdir(root_dir)
    elif Path.cwd().resolve() != root_dir:
        raise RuntimeError(
            "mock_snakemake has to be run from the repository root"
            f" {root_dir} or scripts directory {script_dir}"
        )
    try:
        for p in sm.SNAKEFILE_CHOICES:
            if os.path.exists(p):
                snakefile = p
                break
        kwargs = (
            dict(rerun_triggers=[]) if parse(sm.__version__) > Version("7.7.0") else {}
        )
        if isinstance(configfiles, str):
            configfiles = [configfiles]

        workflow = sm.Workflow(snakefile, overwrite_configfiles=configfiles, **kwargs)
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
                for i in range(len(io)):
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
        timezone = pytz.timezone(pytz.country_timezones[node[:2]][0])
        tz_dt_index = dt_index.tz_convert(timezone)
        week_df[node] = [24 * dt.weekday() + dt.hour for dt in tz_dt_index]
        week_df[node] = week_df[node].map(weekly_profile)

    week_df = week_df.tz_localize(localize)

    return week_df


def parse(l):
    if len(l) == 1:
        return yaml.safe_load(l[0])
    else:
        return {l.pop(0): parse(l)}


def update_config_with_sector_opts(config, sector_opts):
    from snakemake.utils import update_config

    for o in sector_opts.split("-"):
        if o.startswith("CF+"):
            l = o.split("+")[1:]
            update_config(config, parse(l))


def get_technology_mapping(
    fn,
    source: str = "REMIND-EU",
    target: str = "PyPSA-EUR",
    flatten: bool = False,
) -> dict:
    """
    Get a mapping between technologies in REMIND and PyPSA-EUR.

    Some technologies may be mapped to generalised technologies.
    Valid values for source and target are: remind, general, pypsa-eur.
    Lower and uppercase are ignored.

    Parameters
    ----------
    fn : str
        Path to the technology mapping file.
    source : str, optional
        Technology mapping source.
        Valid values are: PyPSA-EUR, General, REMIND-EU.
        Default "REMIND-EU".
    target : str, optional
        Technology mapping target.
        Valid values are: PyPSA-EUR, General, REMIND-EU.
        Default "PyPSA-EUR".
    flatten : bool, optional
        Whether to try to flatten the mapping; only valid
        if the mapping is unique for all keys.
        Default False.

    Returns
    -------
    dict
        Dictionary with source technologies as keys and a list of target technologies (flatten = False) or a single element (flatten = True) as values.
    """
    # Group by source first and aggregate targets to list
    # as targets could be one or more technologies which would
    # vanish if we just convert it to a dict
    df = (
        pd.read_csv(fn)
        .groupby(source.lower())[target.lower()]
        .apply("unique")
        .apply(list)
    )

    if flatten:
        if (df.apply(lambda x: len(x)) != 1).any():
            logger.error(f"Cannot flatten mapping. Non-unique map contained:\n {df}")

        df = df.apply(lambda x: x[0])

    return df.to_dict()


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


def read_remind_data(file_path, variable_name, rename_columns={}):
    """
    Auxiliary function for standardised reading of REMIND-EU data files to
    pandas.DataFrame.

    Here all values read are considered variable, i.e. use
    "variable_name" also for what is considered a "parameter" in the GDX
    file.
    """
    import re

    from gams import transfer as gt

    remind_data = gt.Container(file_path)

    data = remind_data[variable_name]
    df = data.records

    if df is not None and not df.empty:
        # Hack to make weird column naming with GAMS API <= 42 comptaible with >= 43
        # where columns where always numbered with "_<index>" even if no duplicate columns were present
        # but we want to keep duplicate columns differentiation with "_1" and "_2" if columns with same names are present,
        # e.g. for "pe2se" with two columns named "all_enty"
        if max({i: data.domain.count(i) for i in data.domain}.values()) == 1:
            df.columns = data.domain + list(
                df.columns[len(data.domain) :]
            )  # Preserve all remaining column names, espc. "value" or "level" column name for parameters or variables
    else:
        # Handle empty records by creating an empty DataFrame to return

        # assign last_column name based on the datatype of data
        if isinstance(data, gt.Parameter):
            last_column_name = "value"
        elif isinstance(data, gt.Variable):
            last_column_name = "level"

        df = pd.DataFrame(columns=data.domain + [last_column_name])

    df = df.rename(columns=rename_columns, errors="raise")

    return df
