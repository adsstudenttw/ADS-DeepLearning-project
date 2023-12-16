# `show_logs` target: Run the MLflow server to visualize experiment logs
# Start the MLflow server with the specified configuration
# Set the URI for the backend store (where MLflow metadata is stored)
# Set the default root directory for storing artifacts (e.g., models, plots)
# Set the host for the MLflow server to bind to (localhost in this case)
show_logs:
	mlflow server \
		--backend-store-uri sqlite:///mlflow.db \
		--default-artifact-root ./mlruns \
		--host 127.0.0.1
	
# `stop_server` target: Check if an MLflow server is running on port 5000 and shut it down if it is
# Find the process listening on port 5000, filter by 'mlflow', extract its process ID, and terminate it
stop_server:
	@-lsof -i :5000 -sTCP:LISTEN | grep 'mlflow' | awk '{ print $$2 }' | xargs -I {} kill {}

.PHONY: help install build test lint format

.DEFAULT: help
help:
	@echo "make lint"
	@echo "       run flake8 and mypy"
	@echo "make format"
	@echo "       run isort and black"
	@echo "make help"
	@echo "       print this help message"

lint:
	pdm run ruff ../notebook --fix
	pdm run mypy ../notebook  --ignore-missing-imports

format:
	pdm run black ../notebook