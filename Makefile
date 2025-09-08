.PHONY: install test streamlit backtest

install:
	pip install -e .

test:
	pytest -q

streamlit:
	streamlit run ui/streamlit_app.py

backtest:
	python -m engine.eval.backtester
