echo "------------"
echo "Formatting"
echo "------------"
poetry run isort --profile=black .
poetry run black .

echo "------------"
echo "Linting"
echo "------------"

echo "Linting prepare_data/ ..."
poetry run pylint prepare_data/

echo "Linting retrieval_model_training/ ..."
poetry run pylint retrieval_model_training/

echo "Linting tests/ ..."
poetry run pylint tests/

echo "Linting utils/ ..."
poetry run pylint utils/

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
