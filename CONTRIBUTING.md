# Contributing <!-- omit in toc -->

Basic guidelines for contributing to the project.

## Table of Contents <!-- omit in toc -->

- [Get Started](#get-started)
- [Adding new models](#adding-new-models)

## Get Started

Ready to contribute? Here's how to set up for local development.
Please note this documentation assumes you already have
`uv` and `git` installed and ready to go. If you don't have `uv` installed it be when running
`make install`.

> [!NOTE] General code guidelines
>
> 1. Please make sure you use `ruff` to handle linting and formatting issues. This will ensure that your code is consistent with the rest of the project. This can be done either using the recipe `make check` or by running `ruff check` and `ruff format` directly.
> 2. We use `pytest` for testing. When adding new features or fixing bugs, please make sure to add tests for your changes.
>       - Make sure that your changes pass all tests before submitting a pull request. This can be done either using the recipe `make test` or by running `pytest` directly.
> 3. Please ensure your code is well-documented. This includes docstrings for functions and classes, as well as comments where necessary. Although we do not use type checking via `mypy`, it is **strongly recommended** to add type hints to your code.

1. Clone the repository to your local machine:

    ``` bash
    git clone https://github.com/hssn-20/nucleotide-model-experiments
    ```

2. Now we need to install the environment. If you're using `pyenv` or `conda`, set the active python version to 3.10. Then, install the environment with:

    ``` bash
    make install
    ```

3. Create a branch for local development:

    ``` bash
    git checkout -b name-of-your-bugfix-or-feature
    ```

4. When you're done making changes, run `make check` to ensure that your changes pass all tests and linting checks. After fixing any issues, run `make test` to ensure that test pass.

    ``` bash
    make check # make changes if needed
    make test
    ```

5. Commit your changes and push your branch to GitHub:

    ```bash
    git add .
    git commit -m "Your detailed description of your changes."
    git push origin name-of-your-bugfix-or-feature
    ```

6. Submit a pull request through the GitHub website.

When using `uv`, the `requirements.txt` acts as a lockfile. This means that you should not update it manually.
If you need to add a new package, you should add it to the "dependencies" section of the `pyproject.toml` file and run:

```bash
uv pip compile -o requirements.txt pyproject.toml --no-build-isolation
```

> [!NOTE] Note
> Make sure you use the `--no-build-isolation` flag when running `uv pip compile` and `uv pip install`. Otherwise you will run into build issues (due to the `flash-attn` package used by `evo-model`).

## Adding new models

When adding new models for experimentation, please follow the guidelines:

1. If using custom code, add a module to the `models` directory with the model implementation. For example, if you're adding a new model called `MyModel`, create a file called `my_model.py` in the `models` directory.
   - All models must have a `num_parameters` method that returns the number of parameters in the model. This is used for logging and debugging purposes.
2. Add a corresponding `load_mymodel` function to `utils/model_utils.py` . This function should return an instance of the model class.
3. Modify the `load_model` function in `utils/trainer_utils.py` to include the new model. The key should be a simple name for the model (e.g., `"my_model"`) and the value should be the function name from step 2. For example, if your function is called `load_mymodel`, the dictionary should look like this:

    ```python
    def load_model(model_name: str, **kwargs) -> nn.Module:
        models = {
            "my_model": load_mymodel,
            ...
        }
        return models[model_name](**kwargs)
    ```
