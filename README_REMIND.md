# Installation 

* Install micromamba, see [documentation for details](https://mamba.readthedocs.io/en/latest/installation.html#micromamba) including adding micromamba to your `.bashrc`
* Create micromamba env for PyPSA-EUR (`environment.yaml` file comes from PyPSA-EUR repository, located in `pypsa-eur/envs/`):
```
source /home/jhampp/software/micromamba_1.4.2/etc/profile.d/micromamba.sh
micromamba create --file envs/environment.yaml
```
* GAMS PYthon API

Inside the existing PyPSA-EUR conda env, install the GAMS Python API.
You need the path for the current gams installation.
Given the changes in the GAMS API over the recent versions, it is recommend you use GAMS >= 42.X API for Python.

Check available gams versions
```
module avail gams
```
Load highest version, e.g. `gams/43.4.1`
```
module load gams/43.4.1
```
Find path to GAMS directory:
```
dirname $(which gams)
```
Yielding something like `/p/system/packages/gams/43.4.1`.
Activate PyPSA-EUR environment to install GAMS API in there:
```
micromamba activate pypsa-eur
```
Install API, substitute the GAMS path according to point to the same base directory where `dirname $(which gams)` points to:
```
pip install gams --find-links /p/system/packages/gams/43.4.1/api/python/bdist
```
Details see here: https://www.gams.com/latest/docs/API_PY_GETTING_STARTED.html

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
* minimum capacities for generators are determined from REMIND per, in `import_REMIND_RCL_p_nom_limits.py`
    * remind region
    * aggregate of technologies "general_technology"
    * the minimum capacities are then enforced with a >= constrained in `solve_network.py` and the new options for `{opts} = RCL`

## Changes to config.yaml (incomplete; TODO: update!)

* Increas solar potential
    ```
    config["renewable"]["solar"]["capacity_per_sqkm"] = 1.7 (old) -> 5.1 (new)
    ```
    Reason: Limits to maximum potential were making model in some situations where REMIND-EU wanted to have a higher than permissible build-out of PV in the model.
    The original value of 1.7 was with 1% land availability, the new value represents 3% land availability, following the estimate logic also used in the ENSPRESSO dataset by JRC.