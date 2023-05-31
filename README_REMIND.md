# Installation 

* GAMS PYthon API

Inside conda env
```
pip install gams --find-links /opt/gams/gams43.3_linux_x64_64_sfx/api/python/bdist
```

substitute `/opt/gams/...` with correct path to GAMS directory, e.g. `dirname $(which gams)`

Details see here: https://www.gams.com/latest/docs/API_PY_GETTING_STARTED.html
