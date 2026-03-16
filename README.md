## Pre-requisites

Add a file called `.env` at this level with the following content:

```env
OAUTH_CLIENT_ID=...
OAUTH_CLIENT_SECRET=...
OPEN_AI_PROXY_USER=...
OPEN_AI_PROXY_PASSWORD=...
```

### TL;DR
```bash
uv venv --python 3.13
source .venv/bin/activate

Clean uv cache           :   uv clean
Build a wheel            :   uv build

Run a script from
  - working tree         :   uv run aio_bot
  - a wheel              :   uvx --from dist/aio-0.1.0-py3-none-any.whl aio_bot      
  - project in directory :   uvx --find-links dist --from aio aio_bot

Install a wheel          :   uv pip install dist/aio-0.1.0-py3-none-any.whl
Uninstall a wheel        :   uv pip uninstall aio
```

### Install UV
We recommend installing uv globally with pip:

```
pip install uv
```

On some managed MacOS systems an attempt to install Python packages globally causes the below error:
```
This environment is externally managed
╰─> To install Python packages system-wide, try brew install
    xyz, where xyz is the package you are trying to
    install.
```

The alternatives are:
- Install uv with ```brew install uv```;
- Install uv in a virtualenv (and then make sure that virtualenv is active before using uv);
- Use the ```--break-system-packages``` option: ```pip3 install uv --break-system-packages```;
- If you have conda, install uv in ```$ANACONDA_HOME/bin``` by running ```conda install conda-forge::uv```.

### Create a virtual environment.

```
python -m venv .venv
source .venv/bin/activate    # if *nix
.venv\Scripts\activate       # if Windows  

(or)

uv venv --python 3.13
source .venv/bin/activate    # if *nix
.venv\Scripts\activate       # if Windows
```

### Install Build Tools.
```
pip install setuptools wheel build
```

### Build, Install and Run the Python Package

1) Build your package.
 ```
 cd aio
 uv build

 
 (or)
 
 
 python -m build


 (or)


 pip install -e .  # Install in editable mode for development
 ```

2) Install your package.

 ```
 uv pip install dist/aio-0.1.0-py3-none-any.whl   # to install from the built wheel
 uv pip uninstall aio   # to uninstall the installed package.
 ```

3) Run the ai orchestrator.
 ```
 aio_bot
 ```

## For distribution to others, publish it to the Python Package Index (PyPI)

1) Python Package With ```devpi``` Index
```
pip install devpi devpi-server devpi-client
initialize        : devpi-init
start server      : devpi-server --host 0.0.0.0 --port 3141
devpi use http://0.0.0.0:3141/
login             : devpi login root --password ''
add index         : devpi index -c dev bases=root/pypi
use index         : devpi use root/dev
```

2) Upload package to ```devpi``` Index
```
upload package:
    devpi upload dist/*
Uploading to a specific index (if not using devpi use):
    devpi upload --index <user>/<index_name>
    devpi upload --index root/dev

(or)

twine upload --repository-url http://0.0.0.0:3141/root/dev dist/* -u root -p ''
```

(or)

```
twine upload -r devpi dist/* --verbose
vi ~/.pypirc:
[distutils]
index-servers = devpi

[devpi]
repository = http://localhost:3141/root/dev
username = root
```

3) Install package from ```devpi``` Index

```
uvx --index http://0.0.0.0:3141/root/dev/+simple/ aio@latest
uvx --index http://0.0.0.0:3141/root/dev/ aio
uv tool run --index http://0.0.0.0:3141/root/dev/ aio

(or)

export PIP_INDEX_URL=http://localhost:3141/root/dev/
uvx aio

(or)

pip install -i http://localhost:3141/root/pypi/+simple/ your-package-name
e.g., pip install -i http://localhost:3141/root/dev/+simple/ aio --verbose
```

### UV Notes:

When you run below commands from workspace root, you will observe errors outlined below:

1) ```uvx dist/aio-0.1.0-py3-none-any.whl```
Installed 120 packages in 878ms
An executable named ```aio``` is not provided by package aio.
The following executables are available:

    ```- aio_bot```


Use uvx --from aio <EXECUTABLE-NAME> instead.

**Resolution:**
uvx dist/aio-0.1.0-py3-none-any.whl installs the wheel, then (because no command name follows) tries to execute a console script named ```aio``` (the project name). That script doesn’t exist in the wheel, so uvx prints the reminder that only ```aio_bot``` is exposed.
Tell uvx which script you want: e.g. ```uvx --from dist/aio-0.1.0-py3-none-any.whl aio_bot```.
If you’re already in the project tree, an easier option is ```uv run aio_bot```; uv will reuse the local project rather than reinstalling from the wheel.

2) ```uvx --from aio aio_bot```
  × No solution found when resolving tool dependencies:
  ╰─▶ Because there are no versions of aio and you require aio, we can conclude that your requirements are unsatisfiable.

**Resolution:**
uvx --from aio … tells uv to resolve a published package literally named ```aio```. PyPI (and any other configured index) has no distributions with that name, so dependency resolution stops with “no versions of ```aio```”.
The repo you’re in is just a local project; uvx doesn’t look at your working tree. To run its console entry points, stay in the project root and use ```uv run aio_bot``` (uv loads the current project’s environment) or fall back to ```python -m aio.core.cli``` if you prefer the module form.
If you really need the uvx workflow, you’d first have to make the project installable somewhere uv can reach —- e.g., build a wheel (```uv build```), then ```uvx --find-links dist --from aio aio_bot```, or publish the package to an index under the name ```aio```.
