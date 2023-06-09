# Installation 

* GAMS PYthon API

Inside conda env
```
pip install gams --find-links /opt/gams/gams43.3_linux_x64_64_sfx/api/python/bdist
```

substitute `/opt/gams/...` with correct path to GAMS directory, e.g. `dirname $(which gams)`

Details see here: https://www.gams.com/latest/docs/API_PY_GETTING_STARTED.html

* Setup CPLEX

For now, add CPLEX bin directory to path

* Install micromamba and set environment directory accordingly (see instructions by @euronion elsewhere)

* regional mapping: currently stored in `config/regionmapping_21_EU11.csv` is mapping for REMIND-EU, used to map between PyPSA-EUR countries and REMIND-EU regions. Location configured via Snakefile.