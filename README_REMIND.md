# Installation

* Install micromamba, see [documentation for details](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html). In the desired folder, call:
    ```
    "${SHELL}" <(curl -L https://micro.mamba.pm/install.sh)
    ```
    * If SSL certificates are outdated, ignoring SSL checks might be necessary. Add the `-k` flag to `curl` in this case.
    * Specify the installation location, must not be in a home directory but a path which is guaranteed to be accessible from compute nodes, e.g. `/p/tmp/jhampp/micromamba`
    * Init your shell
    * Configure `conda-forge`
    * Specify a prefix location (the location where environments will be created and stored by default), suggested to use the same folder as for installation location, e.g. `/p/tmp/jhampp/micromamba`

* If you are not installing `micromamba` but only want to use it with environments from someone else: Add micromamba to your `.bashrc` including adding micromamba to your `.bashrc` or alternatively call `micromamba init` from the folder where `micromamba` was installed into
* Create micromamba environment for PyPSA-EUR (`environment.yaml` file comes from PyPSA-EUR repository, located in `pypsa-eur/envs/`). If you want to use CPLEX, check which Python versions are supported. As of writing, only Python<=3.10 is supported, so in the `environment.yaml` file the Python version needs to be restricted before installing the environment by chaning the line
    ```
        - python>=3.8
    ```
    to
    ```
        - python>=3.8,<=3.10
    ```
        and then call mircomamba to create the environment
    ```
        micromamba create -f environment.yaml
    ```
* (Obsolete) GAMS PYthon API: In a previous version it was necessary to install the Python GAMS API manually. In this version `gamspy` is automatically installed and used, making this ancient step finally unnecessary.

* Setup CPLEX: For now, CPLEX is installed in `/p/tmp/jhampp/cplex` and the path modification done in `RunPyPSA.sh` from within the REMIND folder.
    * To install CPLEX (in a new Python environment), e.g. `pypsa-eur`:
    ```
        module purge # avoid conflicting Python versions between micromamba environment and loaded modules
        micromamba activate pypsa-eur
        python /p/tmp/jhampp/cplex/python/setup.py install
    ```
    (maybe alternatively `pip instal cplex` also works instead installing from local `setup.py`? Not tested yet though.)
    The cplex Python API must be installed in the conda environment and possible to import, e.g. in `python -c 'import cplex'` must run without any error shown from within the `pypsa-eur` environment

* regional mapping: currently stored in `config/regionmapping_21_EU11.csv` is mapping for REMIND-EU, used to map between PyPSA-EUR countries and REMIND-EU regions. Location configured via Snakefile.

* technology mapping between REMIND-EU and PyPSA-Eur is inferred based on `config/technology_cost_mapping.csv` and complemented with manual adjustments via `get_technology_mapping(...)` in `scripts/_helpers.py` (e.g. for `offwind-ac`, `offwind-dc` and `ror`, `PHS`)
    * The technology / cost mapping is mainly used to extract costs and other technology data from REMIND
    * It is additionally used for mapping the RCL constraint (pre-investment capacities) from REMIND in PyPSA-Eur
    * It is further used to map back the results from PyPSA-Eur to REMIND-EU in `scripts/extract_coupling_parameters.py`
    * The current setup and file structure is build around the rule `import_REMIND_costs`, which makes using the mapping a bit more complicated for other applications.
    * Changes to the `technology_cost_mapping.csv` file should be made carefully (and results checked). Avoid mappings between REMIND and PyPSA-Eur of nature `m:n`; these mappings work e.g. for lignite and coal, but only as long as all mapped technologies are identical, i.e. lignite and coal map from PyPSA-Eur to exactly the same REMIND-EU technologies.

* Setup Gurobi
    * Installation steps / setup
        * Adding the following line to current shell (should be in .bashrc or in file calling PyPSA-EUR)
        ```
            export SSL_CERT_FILE=/p/projects/rd3mod/ssl/ca-bundle.pem_2022-02-08
            export GRB_CAFILE=/p/projects/rd3mod/ssl/ca-bundle.pem_2022-02-08

            # start gurobi script as before
            export GUROBI_HOME="/home/adrianod/software/gurobi/gurobi1003/linux64"
            export PATH="${PATH}:${GUROBI_HOME}/bin"
            export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"
            export GRB_LICENSE_FILE=/p/projects/rd3mod/gurobi.lic
            export GRB_CURLVERBOSE=1
        ```
        * added `<PyPSA-EUR>/cluster_config/slurm-jobscript.py` which contains the job to run on the compute node created by snakemake.
            There a random port number is created and a ssh tunnel setup to the login node. This jobscript is also added to the `<PyPSA-EUR>/cluster_config/config.yaml`:
        ```
            #!/bin/bash
            # Generate random port number between 42000 and 42424 for jobs
            # to avoid collissions between jobs and reusing same port number for all jobs
            # (causing address already in use errors)
            PORT=$((RANDOM%424 + 42000))
            echo "SSH forwarding for Gurobi WLS access on port: $PORT"

            # set up and use ssh proxy on compute node to get access
            (ssh -N -D $PORT $USER@login01 &);
            export https_proxy=socks5://127.0.0.1:$PORT

            # Standard snakemake job execution
            {exec_job}
        ```
        * An ssh key-pair for connecting from compute nodes to login node needs to be accessible for the user (without password) or a key-pair available in the key chain to access without password
        * Gurobi is installed on cluster and within the python environment using
        ```
            micromamba activate pypsa-eur-gurobi
            micromamba install -c gurobi gurobi
        ```
    * License setup:
        * Requires tunneling to reach Gurobi WLS
        * Script adaptations to tunnel to login node from compute nodes (hard-coded `ssh` tunnel) by Falk
        * Script adaptation requires public-private key pair to be accessible from compute nodes (private key) and login node (corresponding public key), e.g. on login node with your user create a new key-pair using:
        ```
            ssh-keygen -t ed25519 -f ~/.ssh/id_rsa.cluster_internal_exchange -C "$USER@cluster_internal_exchange"
        ```
        and add the contents of `~/.ssh/id_rsa_cluster_internal_exchange.pub` to your `~/.ssh/authorized_keys` file in a new line
        with the appropriate entry in `~/.ssh/config`, e.g.
        ```
        Host login01
            Hostname login01
            User <your username>
            PubKeyAuthentication yes
            IdentitiesOnly yes
            IdentityFile ~/.ssh/id_rsa.cluster_internal_exchange
        ```

# Updating to newest PyPSA-EUR version

(Suggested method for large number of changes; often `git` should automatically be able to merge the two code basis by pulling from `upstream/master`,
but some issues could arise where changes to the configuration / code base were made which are not considered conflicts, thus leading
to silent errors / problems when running the REMIND-coupled PyPSA-EUR version.

* Git stuff:
    * Clone REMIND-coupled PyPSA-Eur into new folder, e.g. `git clone git@github.com:aodenweller/pypsa-eur.git pypsa-eur_v0.10.0`
    * Add PyPSA upstream, `git remote add upstream git@github.com:PyPSA/pypsa-eur.git`
    * Switch to previous branch, e.g. `git checkout remind_develop_pypsa_090` 
    * Create and checkout new branch, e.g. `git checkout -b remind_develop_pypsa_v0.10.0`
    * Fetch from upstream, `git fetch upstream`, this also gets the release tags
    * Merge release into branch, e.g. `git merge v0.10.0`
* Use a programm to compare the two repository folders, e.g. "Meld" or "GitLens" for VSCode and see what changes were made and whether the should be compatible with the code-base changes made for the REMIND-coupling
* Special attention has to be given to the following files (open in compare view side-by-side!)
    * `configs/config.default.yaml` -> configuration changes might be relevant to be transfered into `config/config.remind.yaml`
    * `Snakefile`-> changes might be relevant to be transfered into `Snakefile_remind`
    * Changes to `.smk` files (relevant are those that are included in `Snakefile_remind`): Paths for `resources` and `results` in most cases need to consider wildcards `{scenario}, {year}, {iteration}`. These need to be added to new files which are specific to these wildcards which are introduced as dependencies into `.smk` files
    * `solve_network.py`: Whether `RCL` constraint implementation is still compatible with changes made
* Check release notes

## Cluster configuration

Due to specific nature of PIK cluster, the following resources specific to the cluster are set:
* `partition` and `qos` for rules `extract_coupling_parameter` (which through the checkpoing becomes dependent on internet connection) and all retrieve rules.
    * Using this combination of `partition` and `qos` runs these rules on the cluster login node, thus granting them limited internet access
    * These `qsub` modifiers can be set by CLI for Snakemake (thus keeping them out of the `Snakefile`) or inside the `Snakefile`

## Coupling parameters

### REMIND to PyPSA-Eur

* Dedicated scripts creating input files for PyPSA-EUR from the REMIND export (`REMIND2PyPSAEUR.gdx`)
* Output form is imitating original PyPSA-EUR files input format, trying to thereby create drop-in replacements for the original files
* New files are then used in the PyPSA-EUR rules, by modifying the specified input files (but not the model scripts) in the Snakefile / <rules>.smk filesa
* CO2 price via wildcard, placeholder in config.remind.yaml ; extract prices from REMIND and then create dataframe / file `resources/<scenario>/i<iteration>/co2_price_scnearios.csv` with all combinations of scenarios to be run in PyPSA-EUR

* Preinvestment capacities (for generators) from REMIND are implemented via a constraint ("RCL constraint" = Regional Carrier Limit):
    * preinvestment capacities are determined from REMIND in `import_REMIND_RCL_p_nom_limits.py` (per remind region and technology <-> mapped between REMIND and PyPSA-Eur)
    * The config parameter `everywhere_powerplants` controls which powerplants can be built everywhere (also see below). This is set to all dispatchable powerplants. Only allowing additional generators in locations of existing powerplants (from `powerplantmatching`) can lead to problems if there is e.g. no OCGT generator in France in `powerplantmatching`, but REMIND has a non-zero amount of OCGT in France (real example). Therefore, currently we effectively don't use the locational information from `powerplantmatching` but use PyPSA in a greenfield setting w.r.t. to generators, *not* w.r.t. transmission lines.
    * Additional RCL generators are created (in `add_extra_components.py`) with zero costs (can be build for free); these are required for the constraint to work properly.
    * In `solve_network.py` the RCL constraint is added
    * The ensures that per region and mapped technology, the RCL generators can expand up to the preinvestment capacities passed by REMIND. Only after this limit is exhausted, PyPSA-Eur may choose to build additional capacities (at regular costs) of non-RCL generators.
    * (currently commented out and deactivated in code: the minimum capacities are then enforced with a constraint: `RCL capacities (PyPSA-Eur) <= preinvestment capcaities (REMIND)` in `solve_network.py`)
    * The functionality is enabled with the new wildcard option `{opts} = RCL`
    * The functionality can be configred via the config.yaml file: config["remind_coupling"]["preinvestment_capacities"]
    * The RCL constraint requires `config["electricity"]["everywhere_powerplants"]` to specify all types of carriers to which the RCL constraint should apply, in order for powerplants of every type to be attached to every nodes of the final model.
    * If a carrier is missing here and no pre-existing capacities of that type can be found in a country, e.g. `OCGT` in France today with the current data in `powerplants.csv` from `powerplantmatching`, then the RCL capacity constraint cannot be applied to the model and will be ignored silently (i.e. not preinvestment capacities for `OCGT` would be built in France).
* Costs
    * Currently no regional costs in PyPSA-Eur
    * CO2 costs are extracted before in `determine_co2_scenarios` and implemented via wildcards
    * TODO in future: Implement regional costs, maybe even implement some kind of annualised foresight costs 
* Load
    * Currently only a single load time series that is scaled-up
    * In the future, the gdx parameter `load_price` could also be used for several electricity subsectors (EVs, heat pumps, etc.) 

### PyPSA-Eur to REMIND
Parameters are extracted with a specific snakemake rule and script `extract_coupling_parameters.py`.

Extracted values are aggregated by REMIND region and mapped to REMIND technologies.

* Some mapping is 1:n from PyPSA-EUR:REMIND, see the mapping in the file.
* some extracted parameters are weighted by the installed capacity in *REMIND* in from the run before PyPSA-EUR is called, where the weighing is between the different technologies mapped from 1 PyPSA-EUR tech -> n REMIND techs. Currently weighted are: generation shares, installed capacities

## Changes to config.yaml (incomplete; TODO: update!)

* Increase solar potential
    ```
    config["renewable"]["solar"]["capacity_per_sqkm"] = 1.7 (old) -> 5.1 (new)
    ```
    Reason: Limits to maximum potential were making model in some situations where REMIND-EU wanted to have a higher than permissible build-out of PV in the model.
    The original value of 1.7 was with 1% land availability, the new value represents 3% land availability, following the estimate logic also used in the ENSPRESSO dataset by JRC.
* Capacities of existing powerplants are ignored, i.e. no free existing capacities are built by the model. The key is to set:
    * For conventional powerplants (fossil, nuclear, hydro/ror/PHS): `electricity['conventional_carrier'] = []` but to keep the conventional carriers listed in `electricity['extendable_carrier']['Generator']`
    * For RES: `electricity['renewable_carirer']` list the RES carriers, list them as `electricity['extendable_carriers']['Generator']` as well and set `electricity['estimate_renewable_capacities']['enable']=false`
* Correcting hydro power capacities / `electricity['custom_powerplants']=false`: Hydro power capacities a inconsistent between PyPSA-Eur (taken from powerplantmatching) and REMIND, there is also an inconsistency on what is considered "hydro power" in both models (dam, ror, PHS in PyPSA-Eur) vs. (dam, ror in REMIND-EU). The capacities can be corrected via `data/custom_powerplants.csv`, a try was made for DEU (see entries in the file), but this feature is currently deactivated.
* Automatic time resolution:
    * By setting the time-resolution `opt` wildcard to `0H`, the time resolution is automatically adjusted based on the REMIND iteration PyPSA-Eur is currently in
    * The behaviour is configured via the config file in `remind_coupling['automatic_time_resolution']` where iteration ranges and their respective time resolutions can be configured
    * The behaviour is implemented in the checkpoint rule `determine_co2_price_scenarios`


# Troubleshooting

* Insufficient memory:
    * Adjust this line in `rules/common.smk`
    ```
        return int(factor * (5000 + 195 * int(w.clusters)))
    ```
    to more base memory per `solve_network` rule, default PyPSA-Eur is:
    ```
        return int(factor * (10000 + 195 * int(w.clusters)))
    ```
* `KeyError` when calling `snakemake` with `--dry-run / -n`: This can happen due to `group-components` in `cluster_config/config.yaml` and be ignored. The non-dry-run of `snakemake` should run without issues
* `snakemake` stuck in an endless loop: If running `snakemake` directly from the folder and only partially executing rules, e.g. regenerating some output files, sometimes an endless loop of repeating jobs may be submitted. This is related to a `snakemake` bug with checkpoints and cluster execution. To avoid this one can try to regenerate with the same `snakemake` call also the output of the checkpoint, i.e. `sanekmake [your regular arguments] [your target rule or output] -f results/[your scenario]/[your iteration]/co2_price_scenarios.csv`
* Random filesystem errors with `snakemake`: This can happen on the cluster when the filesystem index is not updated fast enough. Also see [this GitHub issue](Ihttps://github.com/snakemake/snakemake/issues/39).
    * Workaround 1: Add
    ```
        for f in files:
        os.listdir(os.path.dirname(f))
    ```
    to the function `wait_for_files` in  `io.py` in the `snakemake` directory of the environment, e.g. `/p/tmp/adrianod/software/micromamba_20240118/envs/pypsa-eur-20240118/lib/python3.10/site-packages/snakemake`. This needs to be repeated for every new environment.
        * Actually, this seems to lead to another weird issue where `snakemake` keeps resubmitting jobs, although they were finished. 
    * Workaround 2: Open another session and run an infinite loop
    ```
    while :; do ls $OUTDIR ; sleep 10; done
    ```
    in the `pypsa-eur/results/<scenario>/<iteration>` directory