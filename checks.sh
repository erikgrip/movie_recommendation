echo "------------"
echo "Formatting"
echo "------------"
poetry run isort --profile=black .
poetry run black .

echo "------------"
echo "Linting"
echo "------------"

echo "Linting src/ ..."
poetry run pylint src/

echo "Linting tests/ ..."
poetry run pylint tests/

echo "Linting train.py ..."
poetry run pylint train.py

echo "------------"
echo "Type Checking"
echo "------------"

poetry run mypy .

echo "------------"
echo "Testing"
echo "------------"
poetry run pytest tests/ -m "not slow"

echo "------------"
