# One-command entry points. Override the interpreter with: make test PY=python3
PY ?= .venv/bin/python
RUN = PYTHONPATH=. $(PY)

.PHONY: setup test check-data train freeze validate final-test reproduce dashboard dashboard-data

setup:            ## create .venv and install pinned dependencies
	python3 -m venv .venv && .venv/bin/pip install -e ".[test]"

test:             ## run the full test suite
	$(RUN) -m pytest -q

check-data:       ## verify processed data matches the split manifests
	$(RUN) scripts/check_data.py

train:            ## retrain the all-26-lab model and calibrators (validation only)
	$(RUN) experiments/run_all_lab_training_evaluation.py

freeze:           ## select thresholds on validation and freeze the final protocol
	$(RUN) experiments/run_final_evaluation.py --stage freeze

validate:         ## full final evaluation on the validation evaluation half
	$(RUN) experiments/run_final_evaluation.py --stage all --split validation

final-test:       ## locked held-out evaluation (logged in the test-access ledger)
	$(RUN) experiments/run_final_evaluation.py --stage all --split test --locked-test

reproduce: check-data test validate  ## everything except retraining and test access

dashboard-data:   ## export real results + held-out patient replays to dashboard/data.js
	$(RUN) dashboard/build_dashboard_data.py

dashboard: dashboard-data  ## build the data and open the dashboard (offline, no server needed)
	open dashboard/index.html
