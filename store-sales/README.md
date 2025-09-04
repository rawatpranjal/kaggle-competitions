# Store Sales - Time Series Forecasting

Use machine learning to predict grocery sales at Favorita stores in Ecuador.

## Competition Overview

Predict sales for thousands of product families sold at Favorita stores located in Ecuador. This is a time series forecasting problem with 15 days of future sales to predict.

## Dataset Description

### Files
- `train.csv` - Training data with time series features and target sales
- `test.csv` - Test data for prediction (15 days after training end)
- `stores.csv` - Store metadata (city, state, type, cluster)
- `oil.csv` - Daily oil prices (Ecuador is oil-dependent)
- `holidays_events.csv` - Holidays and events with metadata
- `transactions.csv` - Additional transaction data
- `sample_submission.csv` - Submission format

### Key Features
- **store_nbr**: Store identifier
- **family**: Product family/category
- **date**: Date of sales
- **sales**: Total sales (target variable)
- **onpromotion**: Number of items on promotion

### Important Notes
- Wages paid bi-weekly (15th and month-end) affect sales
- April 16, 2016 earthquake significantly impacted sales
- Oil prices affect Ecuador's economy and consumer behavior
- Holiday transfers and bridge days affect shopping patterns

## Approach

This is a time series forecasting problem requiring:
1. **Time series analysis** of sales patterns
2. **External factor integration** (oil prices, holidays, promotions)
3. **Store and product segmentation** analysis
4. **Seasonality and trend modeling**
5. **Feature engineering** from dates and external data

## Folder Structure
```
store-sales/
├── data/           # Competition data files
├── code/           # Analysis and modeling scripts  
├── submissions/    # Model predictions
└── README.md       # This file
```