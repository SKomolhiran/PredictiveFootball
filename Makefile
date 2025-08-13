.PHONY: prepare train predict backtest app mlflow

prepare:
	python -c "print('✅ Repo ready. Create data/raw/matches.csv before training.')"

train:
	python src/train.py --config config/config.yaml

predict:
	python src/predict.py --config config/config.yaml --fixtures data/processed/fixtures.csv

backtest:
	python src/backtest.py --config config/config.yaml --from 2019-01-01 --to 2024-12-31

app:
	streamlit run app/edge_board.py

mlflow:
	mlflow ui
