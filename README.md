### Pre-requisites

# TLDR
uv venv --python 3.13
source .venv/bin/activate

Build a wheel         :   uv build
Test a wheel          :   uvx dist/rgai-0.1.0-py3-none-any.whl

Install a wheel       :   uv pip install dist/rgai-0.1.0-py3-none-any.whl
Uninstall a package   :   uv pip uninstall rgai

# Install UV
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

# Create a virtual environment.
    ```
    python -m venv .venv
    source .venv/bin/activate    # if *nix
    .venv\Scripts\activate       # if Windows  

    (or)
    
    uv venv --python 3.13
    source .venv/bin/activate    # if *nix
    .venv\Scripts\activate       # if Windows
    ```

# Install Build Tools.
   ```
   pip install setuptools wheel build
   ```

### Build, Install and Run the Python Package
1) Build your package.
     ```
     cd multi-agent-configuration
     uv build
     ```
     
     (or)
     
     
     ```
     python -m build
     ```

     (or)

     ```
     pip install -e .  # Install in editable mode for development
     ```

2) Install your package.

     ```
     uv pip install dist/rgai-0.1.0-py3-none-any.whl   # to install from the built wheel
     uv pip uninstall rgai   # to uninstall the installed package.
     ```

3) Run the agent orchestrator.
     ```
     agentorchestrator
     ```

######################################


### For distribution to others, publish it to the Python Package Index (PyPI)

1) Python Package With ```devpi``` Index
```
pip install devpi devpi-server devpi-client
initialize: devpi-init
start server  : devpi-server --host 0.0.0.0 --port 3141
devpi use http://0.0.0.0:3141/
login         : devpi login root --password ''
add index     : devpi index -c dev bases=root/pypi
use index     : devpi use root/dev
```

2) Upload package to ```devpi``` Index
```
upload package: devpi upload dist/*
Uploading to a specific index (if not using devpi use):
                devpi upload --index <user>/<index_name>
e.g.,           devpi upload --index root/dev
```

(or)

```
twine upload --repository-url http://0.0.0.0:3141/root/dev dist/* -u root -p ''
```

(or)

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
uvx --index http://0.0.0.0:3141/root/dev/+simple/ rgai@latest
uvx --index http://0.0.0.0:3141/root/dev/ rgai
uv tool run --index http://0.0.0.0:3141/root/dev/ rgai

(or)

export PIP_INDEX_URL=http://localhost:3141/root/dev/
uvx rgai

(or)

pip install -i http://localhost:3141/root/pypi/+simple/ your-package-name
e.g., pip install -i http://localhost:3141/root/dev/+simple/ rgai --verbose
```
