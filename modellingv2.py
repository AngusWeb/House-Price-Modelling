
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split



# Create DataFrames from the JSON data
df_all = pd.read_json('suffolk_all_data.json')
df_new_homes = pd.read_json('suffolk_new_homes.json')
df_new_homes = df_new_homes.drop_duplicates(['price_json', 'latitude', 'longitude'])
# Add a 'New Home' column to df_all and set it to False initially
df_new_homes['New Home'] = 1

# Merge df_all with df_new_homes based on 'price_json', 'latitude', and 'longitude'
df_merged = pd.merge(df_all, df_new_homes[['price_json', 'latitude', 'longitude', 'New Home']],
                     on=['price_json', 'latitude', 'longitude'],
                     how='left')
df_merged['New Home'] = df_merged['New Home'].fillna(0)
# Find columns present in df_new_homes but not in df_all
missing_columns = set(df_new_homes.columns) - set(df_all.columns)

# Print the missing columns
if missing_columns:
    print("Columns not present in all data from new homes:")
    for column in missing_columns:
        print(column)
else:
    print("All columns from new homes are present in the all data DataFrame.")
    
df = df_merged.copy()


df = df.drop(columns=['postcode','full_postcode','date_added','number_bedrooms'])
df['bathrooms'] = df['bathrooms'].fillna(0)
print(df.isna().sum())
df = df.dropna()
print(df.columns)
#encoding
le = LabelEncoder()

# Fit and transform the 'type' column
df['type_encoded'] = le.fit_transform(df['propertySubType'])
#df['branch_encoded'] = le.fit_transform(df['branchDisplayName'])
print(df[['propertySubType', 'type_encoded']])

df = pd.get_dummies(df, columns=['New Home'])
df = df.rename(columns={'New Home_0.0': 'New Home_0', 'New Home_1.0': 'New Home_1'})
df['bedrooms_new_home'] = df['bedrooms'] * df['New Home_1']
df['bathrooms_new_home'] = df['bathrooms'] * df['New Home_1']

df_x = df[['bedrooms','bathrooms','numberOfImages','numberOfFloorplans','numberOfVirtualTours','latitude','longitude','type_encoded','New Home_0','New Home_1','bedrooms_new_home','bathrooms_new_home']]
df_y = df['price_json']

X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=42)

from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd

# Combine your features and target into a single DataFrame for training
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

# Convert to TabularDataset
train_data = TabularDataset(train_data)
test_data = TabularDataset(test_data)

# Define the label column
label = 'price_json'

# Train the model
predictor = TabularPredictor(label=label).fit(train_data)

# Make predictions on the test set
y_pred = predictor.predict(test_data)

# Evaluate the model
performance = predictor.evaluate(test_data)
print(f"AutoGluon Model Performance: {performance}")

# If you want to see more detailed information about the models
leaderboard = predictor.leaderboard(test_data)
print(leaderboard)

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
print(f"R-squared score of the best model: {r2}")

for model_name in predictor.get_model_names():
    model_predictions = predictor.predict(test_data, model=model_name)
    model_r2 = r2_score(y_test, model_predictions)
    print(f"R-squared score of {model_name}: {model_r2}")
    
    # Load the new data
new_houses_df = pd.read_json('suffolkrecenthouselistings_newhome.json')


# Encode the propertySubType
new_houses_df['type_encoded'] = le.transform(new_houses_df['propertySubType'])

# Create dummy variables for 'New Home'
new_houses_df = pd.get_dummies(new_houses_df, columns=['New Home'])

# Ensure all necessary columns are present
required_columns = ['bedrooms', 'bathrooms', 'numberOfImages', 'numberOfFloorplans', 'numberOfVirtualTours', 
                    'latitude', 'longitude', 'type_encoded', 'New Home_0', 'New Home_1','bedrooms_new_home','bathrooms_new_home']

for col in required_columns:
    if col not in new_houses_df.columns:
        new_houses_df[col] = 0  # or another appropriate default value

# Select the features
new_houses = new_houses_df[required_columns]

print(new_houses)

# Make predictions using the AutoGluon predictor
predicted_prices = predictor.predict(new_houses)

# Add predictions to the DataFrame
new_houses_df['predicted_price'] = predicted_prices

# Display the predicted prices alongside the actual pautorices
result = new_houses_df[['predicted_price', 'price_json']]
print(result)

# Calculate and print some error metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

mae = mean_absolute_error(new_houses_df['price_json'], predicted_prices)
rmse = np.sqrt(mean_squared_error(new_houses_df['price_json'], predicted_prices))
mape = np.mean(np.abs((new_houses_df['price_json'] - predicted_prices) / new_houses_df['price_json'])) * 100

print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Squared Error: {rmse}")
print(f"Mean Absolute Percentage Error: {mape}%")