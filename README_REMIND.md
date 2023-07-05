# Installation 

* GAMS PYthon API

Inside the existing PyPSA-EUR conda env, install the GAMS Python API:

```
pip install gams --find-links /opt/gams/gams43.3_linux_x64_64_sfx/api/python/bdist
```

substitute `/opt/gams/...` with correct path to GAMS directory, e.g. `dirname $(which gams)`

Details see here: https://www.gams.com/latest/docs/API_PY_GETTING_STARTED.html

* Setup CPLEX

For now, add CPLEX bin directory to path

* Install micromamba and set environment directory accordingly (see instructions by @euronion elsewhere)

* regional mapping: currently stored in `config/regionmapping_21_EU11.csv` is mapping for REMIND-EU, used to map between PyPSA-EUR countries and REMIND-EU regions. Location configured via Snakefile.


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