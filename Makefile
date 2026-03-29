.PHONY: install
install: ## Install the virtual environment and install the pre-commit hooks
	@echo "🚀 Creating virtual environment using pixi"
	@pixi install
	@pixi run pre-commit install

.PHONY: check
check: ## Run code quality tools.
	@echo "🚀 Linting code: Running pre-commit"
	@pixi run pre-commit run -a
	@echo "🚀 Static type checking: Running pyright"
	@pixi run pyright
	@echo "🚀 Checking for obsolete dependencies: Running deptry"
	@pixi run deptry .

.PHONY: test
test: ## Test the code with pytest
	@echo "🚀 Testing code: Running pytest"
	@pixi run python -m pytest --cov --cov-config=pyproject.toml --cov-report=xml

.PHONY: download-test-files
download-test-files: ## Download samples to run tests
	@echo "🚀 Downloading test files"
	@pixi run python tests/download_test_files.py

.PHONY: build
build: clean-build ## Build wheel file
	@echo "🚀 Creating wheel file"
	@pixi run python -m build

.PHONY: clean-build
clean-build: ## Clean build artifacts
	@echo "🚀 Removing build artifacts"
	@pixi run python -c "import shutil; import os; shutil.rmtree('dist') if os.path.exists('dist') else None"

.PHONY: publish
publish: ## Publish a release to PyPI.
	@echo "🚀 Publishing."
	@pixi run twine upload --repository-url https://upload.pypi.org/legacy/ dist/*

.PHONY: build-and-publish
build-and-publish: build publish ## Build and publish.

.PHONY: docs-test
docs-test: ## Test if documentation can be built without warnings or errors
	@pixi run mkdocs build -s

.PHONY: docs
docs: ## Build and serve the documentation
	@pixi run mkdocs serve

.PHONY: docs-publish
docs-publish: ## Publish documentation to GitHub Pages
	@pixi run mkdocs gh-deploy

.PHONY: api
api: ## Start the API server with uvicorn (production)
	@echo "🚀 Starting AgriGEE.lite API"
	@pixi run -e api agl_api --host 0.0.0.0 --port 8000

.PHONY: api-debug
api-debug: ## Start the API in debug mode with fastapi dev (hot-reload, no uvicorn)
	@echo "🚀 Starting AgriGEE.lite API in debug mode"
	@pixi run -e api fastapi dev agrigee_lite/api/_app.py --host 127.0.0.1 --port 8000

.PHONY: help
help:
	@pixi run python -c "import re; \
	[[print(f'\033[36m{m[0]:<20}\033[0m {m[1]}') for m in re.findall(r'^([a-zA-Z_-]+):.*?## (.*)$$', open(makefile).read(), re.M)] for makefile in ('$(MAKEFILE_LIST)').strip().split()]"

.DEFAULT_GOAL := help
