# Normalization Techniques for Machine Learning in Norwegian Stock Market Prediction

## Project Overview

This research project investigates the impact of different data normalization techniques on machine learning models for predicting directional movements in the Norwegian stock market (OSEBX). The study evaluates five normalization methods across two model architectures to determine their effects on both prediction performance and consistency.

### Research Question

What is the impact of different data normalization techniques on the performance of machine learning models for stock price prediction in the Norwegian stock market?

## Key Features

- Systematic evaluation of 5 normalization techniques: min-max, z-score, median, sigmoid, and hyperbolic tangent estimator
- Comparison across tree-based (LightGBM) and neural network (LSTM) model architectures
- Time-series cross-validation with 10 folds to ensure robust evaluation
- Statistical significance testing using Wilcoxon signed-rank tests
- Comprehensive analysis of both performance metrics and prediction consistency

## Dataset

- Target: Oslo Stock Exchange Benchmark Index (OSEBX.OL)
- Time period: January 2014 to December 2024
- Features: Engineered from OHLCV (Open, High, Low, Close, Volume) data
- Technical indicators: Price returns, moving averages, volatility measures, RSI, MACD, Bollinger Bands, etc.
- Target variable: Binary classification of price direction (up/down) after 5 days

## Methodology

1. **Data Collection**: Historical OSEBX data retrieved via Yahoo Finance API
2. **Feature Engineering**: Time-aware approach to prevent data leakage
3. **Cross-Validation**: 10-fold time-series splits with 5-day gap between train/test
4. **Normalization Comparison**: Each normalization technique applied independently
5. **Model Training**: Optimized LightGBM and LSTM models for each normalization method
6. **Statistical Analysis**: Wilcoxon tests to evaluate significance of performance differences

## Key Findings

1. **Differential Impact**: Normalization techniques affect neural networks more significantly than tree-based models
2. **Consistency Advantage**: Sigmoid and tanh estimator normalizations substantially reduced prediction variance
3. **Significant Degradation**: Z-score and median normalization showed statistically significant performance degradation for LSTM models
4. **Model Architecture**: LSTM models consistently outperformed LightGBM regardless of normalization technique
5. **Market Efficiency**: Overall modest predictive performance suggests efficiency in the Norwegian market

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/norwegian-stock-normalization.git
cd norwegian-stock-normalization

# Installation if you have uv installed
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# Or using pip
pip install -r requirements.txt
```

## Project Structure

```
├── data/                  # Data cache directory
├── results/               # CSV output files with evaluation metrics
├── visualizations/        # Generated plots and visual analysis
├── src/
│   ├── main.py            # Main execution script
│   ├── normalization.py   # Normalization implementation
│   ├── feature_eng.py     # Feature engineering functions
│   ├── models.py          # Model definitions (LightGBM, LSTM)
│   └── utils.py           # Utility functions
├── notebooks/             # Jupyter notebooks for exploration
├── requirements.txt       # Project dependencies
└── README.md              # This file
```

## Dependencies

- Python 3.12+
- pandas
- numpy
- scikit-learn
- LightGBM
- PyTorch
- matplotlib
- seaborn
- yfinance
- optuna
- scipy
- tqdm

## Citation

If you use this code or findings in your work, please cite:

```
@article{normalization_norwegian_stock,
  title={Normalization in Financial Machine Learning: Comparative Analysis of Techniques for Norwegian Market Prediction},
  author={},
  journal={},
  year={2025},
  volume={},
  pages={}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Acknowledge any funding sources
- Computational resources used
- Any individuals who provided feedback or assistance
