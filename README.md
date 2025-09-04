# Kaggle Competitions

Machine learning competition implementations and analysis.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
chmod 600 ~/.kaggle/kaggle.json
```

## Competitions

### Titanic - Machine Learning from Disaster
**Type:** Binary Classification  
**Location:** `titanic/`

Binary classification to predict passenger survival. Implemented comprehensive feature engineering including family relationships, interaction terms, and saturated logistic models. Used L1 regularization for feature selection and compared multiple algorithms.

**Results:**
- Cross-validation: 82.5% accuracy
- Best approach: 4-feature model (Sex_Male, Age_Child, CabinCount, LargeFamily)

### House Prices - Advanced Regression Techniques  
**Type:** Regression  
**Location:** `house-prices/`

Predict house sale prices using 79 features. Extensive analysis included correlation studies, feature engineering, transformation testing, and hyperparameter optimization. Implemented proper cross-validation methodology to prevent data leakage.

**Results:**
- Leaderboard score: 0.12241 RMSE
- Best approach: CatBoost + Box-Cox transformation + 116 engineered features
- Key finding: Box-Cox (λ=-0.077) outperformed standard log transformation

### Store Sales - Time Series Forecasting
**Type:** Time Series Forecasting  
**Location:** `store-sales/`

Predict 15-day sales forecast for grocery stores in Ecuador. Dataset contains 3M+ records across 54 stores and 33 product families with external factors (oil prices, holidays, promotions).

**Status:** Data exploration complete
- 4+ years of daily sales data (2013-2017)
- Strong seasonality and promotion effects identified
- 619% sales lift from promotions