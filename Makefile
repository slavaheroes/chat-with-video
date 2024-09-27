# Define variables
STREAMLIT_SERVICE = streamlit

setup: # install pre-commit hooks
	pip install pre-commit black isort pylint
	pre-commit install
	pre-commit run --all-files

attach-app:
	$ sudo docker exec -it $(STREAMLIT_SERVICE) /bin/bash

compose-down:
	$ sudo docker-compose down -v

compose-up:
	$ sudo docker-compose up -d

compose-up-force:
	$ sudo docker-compose up -d --build --force-recreate
