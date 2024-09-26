

setup: # install pre-commit hooks
	pip install pre-commit black isort pylint
	pre-commit install
	pre-commit run --all-files