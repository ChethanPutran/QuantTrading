.PHONY: test run lint

test:
	.venv/bin/python -m pytest -q

run:
	.venv/bin/python main.py

lint:
	.venv/bin/python -m compileall src main.py main_async.py