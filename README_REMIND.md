# Installation 

* Install micromamba, see [documentation for details](https://mamba.readthedocs.io/en/latest/installation.html#micromamba) including adding micromamba to your `.bashrc`
* Create micromamba env for PyPSA-EUR (`environment.yaml` file comes from PyPSA-EUR repository, located in `pypsa-eur/envs/`):
```
source /home/jhampp/software/micromamba_1.4.2/etc/profile.d/micromamba.sh
micromamba create --file envs/environment.yaml
```
* GAMS PYthon API: In a previous version it was necessary to install the Python GAMS API manually. In this version `gamspy` is automatically installed and used, making this ancient step finally unnecessary.

* Setup CPLEX
For now, add CPLEX bin directory to path, in `~/.bashrc` add the following line:
```
export PATH=/home/adrianod/software/cplex/cplex/bin/x86-64_linux:$PATH
```
and install the Python package into the `pypsa-eur` environment:
```
micromamba activate pypsa-eur
python /home/adrianod/software/cplex/python/setup.py install
```

(maybe alternatively `pip instal cplex` also works instead installing from local `setup.py`? Not tested yet though.)

* Install micromamba and set environment directory accordingly (see instructions by @euronion elsewhere)
* The cplex Python API must be installed in the conda environment and possible to import, e.g. `python -m 'import cplex'` must be possible in the conda environment

* regional mapping: currently stored in `config/regionmapping_21_EU11.csv` is mapping for REMIND-EU, used to map between PyPSA-EUR countries and REMIND-EU regions. Location configured via Snakefile.
* technology mapping stored in `config/technology_mapping.csv` used by multiple rules

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
to silend errors / problems when running the REMIND-coupled PyPSA-EUR version.

* Download REMIND-coupled PyPSA-EUR and original (upstream) version of PyPSA-EUR you want to update to into separate folders
* Use a programm to compare the two repository folders, e.g. "Meld" or "GitLens" for VSCode and see what changes were made and whether the should be compatible with the code-base changes made for the REMIND-coupling
* Special attention has to be given to the following files:
    * `configs/config.default.yaml` -> configuration changes might be relevant to be transfered into `config/config.remind.yaml`
    * `Snakefile`-> changes might be relevant to be transfered into `Snakefile_remind`
    * Changes to `.smk` files: Paths for `resources` and `results` in most cases need to consider wildcards `{scenario}, {year}, {iteration}`. These need to be added to new files which are specific to these wildcards which are introduced as dependencies into `.smk` files
    * `solve_network.py`: Whether `RCL` constraint implementation is still compatible with changes made

## Cluster configuration

Due to specific nature of PIK cluster, the following resources specific to the cluster are set:
* `partition` and `qos` for rules `extract_coupling_parameter` (which through the checkpoing becomes dependent on internet connection) and all retrieve rules.
    * Using this combination of `partition` and `qos` runs these rules on the cluster login node, thus granting them limited internet access
    * These `qsub` modifiers can be set by CLI for Snakemake (thus keeping them out of the `Snakefile`) or inside the `Snakefile`

## coupling parameters


from PyPSA-EUR -> REMIND are extract with specific snakemake rule and script `extract_coupling_parameters.py`.

Extracted values are aggregated by REMIND region and mapped to REMIND technologies.

* Some mapping is 1:n from PyPSA-EUR:REMIND, see the mapping in the file.
* some extracted parameters are weighted by the installed capacity in *REMIND* in from the run before PyPSA-EUR is called, where the weighing is between the different technologies mapped from 1 PyPSA-EUR tech -> n REMIND techs. Currently weighted are: generation shares, installed capacities


from REMINd -> PyPSA-EUR

* Dedicated scripts creating input files for PyPSA-EUR from the REMIND export ("REMIND2PyPSAEUR.gdx")
* Output form is imitating original PyPSA-EUR files input format, trying to thereby create drop-in replacements for the original files
* New files are then used in the PyPSA-EUR rules, by modifying the specified input files (but not the model scripts) in the Snakefile / <rules>.smk filesa
* CO2 price via wildcard, placeholder in config.remind.yaml ; extract prices from REMIND and then create dataframe / file `resources/<scenario>/i<iteration>/co2_price_scnearios.csv` with all combinations of scenarios to be run in PyPSA-EUR

* preinvestment capacities (for generators) from REMIND are implemented via a constraint ("RCL constraint" = Regional Carrier Limit):
    * preinvestment capacities are determined from REMIND in `import_REMIND_RCL_p_nom_limits.py` (per remind region and technology <-> mapped between REMIND and PyPSA-Eur)
    * In the locations of existing conventional powerplants in PyPSA-Eur, additional generators (clones) are created onto which the `RCL` constraints apply (in `add_extra_components.py`)
    * the minimum capacities are then enforced with a constraint: `RCL capacities (PyPSA-Eur) <= preinvestment capcaities (REMIND)` in `solve_network.py`
    * The functionality is enabled with the new wildcard option `{opts} = RCL`
    * The functionality can be configred via the config.yaml file: config["remind_coupling"]["preinvestment_capacities"]


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