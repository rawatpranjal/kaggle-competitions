# Kaggle Competitions

Repository for working on various Kaggle competitions.

## Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure Kaggle API (place your kaggle.json in ~/.kaggle/)
chmod 600 ~/.kaggle/kaggle.json
```

## Competitions

### 1. Titanic - Machine Learning from Disaster

**Location:** `titanic/`

**Description:** Classic binary classification competition to predict passenger survival on the Titanic.

**Approach:**
- **Model:** LightGBM with early stopping
- **Features:** 
  - Title extraction from names
  - Family size and IsAlone indicators
  - Age and Fare binning
  - Interaction features (Age*Class, Fare_Per_Person)
  - Child and Young_Female indicators
  - HasCabin indicator
- **Preprocessing:**
  - Filled missing values (Age, Fare, Embarked)
  - Label encoding for categorical variables

**Performance:**
- Cross-validation score: 0.8384 (+/- 0.0121)
- **Public Leaderboard Score: 0.72009**

**To Run:**
```bash
cd titanic/code
python train.py
```

**To Submit:**
```bash
kaggle competitions submit -c titanic -f titanic/submission.csv -m "Your message"
```

**Future Improvements:**
- Add more feature engineering (cabin deck extraction)
- Try ensemble methods (stacking, voting)
- Hyperparameter tuning with Optuna
- Cross-validation strategy improvements