# Column configurations for the tourism dataset

# Numeric columns (including ordinal)
NUMERIC_COLS = [
    'Age',
    'DurationOfPitch',
    'MonthlyIncome',
    'NumberOfTrips',   # (low-cardinal) but ordinal
    'NumberOfPersonVisiting', # (low-cardinal) but ordinal
    'NumberOfFollowups', # (low-cardinal) but ordinal
    'PreferredPropertyStar', # (low-cardinal) but ordinal
    'PitchSatisfactionScore', # (low-cardinal) but ordinal
    'NumberOfChildrenVisiting', # (low-cardinal) but ordinal
    'CityTier'  # (low-cardinal) but ordinal
]

# Categorical columns
CATEGORICAL_COLS = [
    'TypeofContact',
    'Occupation',
    'Gender',
    'MaritalStatus',
    'Passport',
    'OwnCar',
    'ProductPitched', # categorical nature (but can posses order) # DOUBTFUL
    'Designation' # categorical nature (but can posses order) # DOUBTFUL
]

# Columns that need ordinal encoding
ORDINAL_COLS = ['Designation', 'ProductPitched']

# Target column
TARGET_COL = 'ProdTaken'

# Eval Metric
TEST_METRIC = 'roc_auc'

MLFLOW_TRACKING_PATH = "mlruns"  # Will be created at root level
