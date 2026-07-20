MAIN=main.py
VENV=venv

PYTHON=$(VENV)/bin/python3
PIP=$(VENV)/bin/pip
FLAKE8=$(VENV)/bin/flake8
MYPY=$(VENV)/bin/mypy

venv:
	python3 -m venv $(VENV)

install: venv
	$(PIP) install -r requirements.txt

run:
	uv run $(PYTHON) -m src

debug:
	$(PYTHON) -m pdb $(MAIN)

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type d -name ".mypy_cache" -exec rm -r {} +
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete

lint:
	$(FLAKE8) .
	$(MYPY) .

lint-strict:
	$(FLAKE8) .
	$(MYPY) . --strict
