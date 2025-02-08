import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Read the combined train and test datasets
train = pd.read_csv("/kaggle/input/ue21cs342aa2/train.csv", index_col=["Date"], parse_dates=['Date'])
test = pd.read_csv("/kaggle/input/ue21cs342aa2/test.csv", index_col=0)

# Train a LightGBM Regressor model for predicting 'Close'
model_close = lgb.LGBMRegressor(n_estimators=100, random_state=42)
model_close.fit(train[['Open', 'Volume']], train['Close'])
predictions_close = model_close.predict(test[['Open', 'Volume']])

# Update the 'Close' values in the test dataset
test['Close'] = predictions_close

# Convert the 'Strategy' classes to numeric labels
label_encoder = LabelEncoder()
train['Strategy'] = label_encoder.fit_transform(train['Strategy'])

# Train a Random Forest Classifier model for predicting 'Strategy'
model_strategy = RandomForestClassifier(n_estimators=100, random_state=42)
model_strategy.fit(train[['Open', 'Close', 'Volume']], train['Strategy'])
predicted_strategy = model_strategy.predict(test[['Open', 'Close', 'Volume']])

# Decode the numeric labels to the original 'Strategy' classes
encoding_dict = {0: 'Buy', 1: 'Sell', 2: 'Hold'}
predicted_strategy = [encoding_dict[prediction] for prediction in predicted_strategy]

test['Strategy'] = predicted_strategy

# Calculate accuracy
accuracy = accuracy_score(test['Strategy'], predicted_strategy)
print(f"Accuracy: {accuracy:.2%}")

# Create the submission DataFrame
submission = pd.DataFrame()
submission["id"] = test.index
submission["Date"] = test["Date"]
submission["Close"] = test["Close"]
submission["Strategy"] = test['Strategy']

# Save the submission DataFrame to a CSV file
submission.to_csv('submission.csv', index=False)

# Display the first few rows of the submission file
print(submission.head())