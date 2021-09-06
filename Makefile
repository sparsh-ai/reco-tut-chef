setup:
	python3 setup.py install

test:
	PYTHONPATH=. pytest

env:
	pip install -q dvc dvc[gdrive]
	pip install -q boto3
	mkdir -p ~/.aws && cp /content/drive/MyDrive/AWS/d01_admin/* ~/.aws