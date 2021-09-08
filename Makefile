setup:
	pip install -e .

test:
	PYTHONPATH=. pytest
