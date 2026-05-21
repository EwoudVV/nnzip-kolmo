# Setting up a development environment

This guide walks through the steps needed to get a working development environment for the project. It assumes you have a recent version of Python installed and are comfortable using a terminal.

## Prerequisites

Before you begin, make sure the following are installed on your machine:

- Python 3.10 or newer
- Git, for cloning the repository and managing changes
- A text editor or IDE of your choice; we recommend VS Code or PyCharm
- Roughly 500 MB of free disk space for dependencies and cached models

If you are on macOS, you may already have Python installed, but we recommend using a version manager such as pyenv or asdf to avoid conflicts with the system Python. On Windows, the official Python installer from python.org is the simplest path.

## Cloning the repository

Open a terminal and navigate to the directory where you would like to keep the project. Then run the following command to clone the repository:

```
git clone https://github.com/example/project.git
cd project
```

This will create a new directory called project containing the source code and supporting files.

## Setting up a virtual environment

We strongly recommend using a virtual environment to isolate the project's dependencies from your system Python. From inside the project directory, create one with:

```
python -m venv .venv
source .venv/bin/activate
```

On Windows, the activation command is slightly different:

```
.venv\Scripts\activate
```

Once the virtual environment is active, your shell prompt should show the name of the environment in parentheses.

## Installing dependencies

The project uses standard Python packaging, so all dependencies can be installed with a single command:

```
pip install -e ".[test]"
```

The square brackets at the end install the optional test dependencies, which you will need if you plan to run the test suite or contribute new tests.

## Running the tests

To verify your setup, run the test suite:

```
pytest -v
```

If all tests pass, you are ready to start working on the project. If any tests fail, check the troubleshooting section below for common issues and their solutions.
