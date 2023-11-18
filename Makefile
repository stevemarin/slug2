
all: ruff isort black test

.PHONY: ruff isort black test

ruff:
	ruff . --fix

isort:
	isort slug2/ tests/

black:
	black .

test:
	coverage run -m pytest .
	coverage html --omit=tests/*
