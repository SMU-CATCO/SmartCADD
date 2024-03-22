# eli-project-template

For dependency management, I make use of [poetry](https://python-poetry.org/). This is a great tool for managing dependencies and virtual environments. I highly recommend using it for your projects. If you are not familiar with poetry, I suggest reading the [documentation](https://python-poetry.org/docs/).

## General Setup

Poetry reccomends using a virtual environment to install Poetry inside of to avoid conflicts and Poetry updates. To do this, run the following:

```bash
conda create -n eli-project-template python=3.11 -y
conda activate eli-project-template
python3 -m pip install --user pipx
python3 -m pipx ensurepath
```

This creates a conda environment called `eli-project-template` and activates it. You can name the environment whatever you want, but I suggest using the name of the project. This way, you can easily see which environment is for which project.

Once you have your environment setup, you can install poetry with the following command:

```bash
pipx install poetry
```

## Project Setup

This project comes with a `pyproject.toml` file. This file is used to manage your project's dependencies and virtual environment. If you want to generate a new one and follow along with the rest of the setup, you can run the following command, or you can just edit the file directly:

```bash
poetry init
```

This will initialize the poetry project and create a `pyproject.toml` file. This file is used to manage your project's dependencies and virtual environment. At this point, you can edit the `pyproject.toml` file to add your dependencies and add your project's information..

You can add your dependencies to the `pyproject.toml` file and then run the following command to install them (this also installs your package so you can use it in your code):

```bash
poetry install --with dev
```

This will create a virtual environment and install your dependencies, including the development group dependencies. This will ensure that everyone is working with the same versions of the same packages. This is great for reproducibility and avoiding conflicts. On SuperPOD, you don't need the `--with-dev` flag, since this is more for MKDocs and other development tools. SuperPOD should be more about just running your code.

Now, edit your `./code_package` to be an appropriate name (and reflect this in you `pyproject.toml`). Once you have that done

## Installing Packages

Instead of using `pip install`, you can use `poetry add` to add a package to your project. This will add the package to your `pyproject.toml` file and install it in your virtual environment. For example, to install `pandas`, you would run the following command:

```bash
poetry add pandas
```

This will add `pandas` to your `pyproject.toml` file and install it in your virtual environment.

## MKDocs

This project uses [MKDocs](https://www.mkdocs.org/) to generate documentation. Under the `docs` directory, you can add markdown files to create your documentation. Once you have added your markdown files, you can run the following command to generate the documentation and serve it locally:

```bash
mkdocs serve
```

For more details on how to use MKDocs, I suggest reading the [documentation](https://www.mkdocs.org/).
