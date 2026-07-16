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
	$(PYTHON) $(MAIN)

debug:
	$(PYTHON) -m pdb $(MAIN)

lint:
	$(FLAKE8) .
	$(MYPY) .

lint-strict:
	$(FLAKE8) .
	$(MYPY) . --strict
