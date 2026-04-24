#! /usr/bin/env bash

# Install pixi
curl -fsSL https://pixi.sh/install.sh | bash

# Install Dependencies
pixi install

# Install pre-commit hooks
pixi run pre-commit install --install-hooks
