import json
import pandas as pd
import folium
from folium.plugins import HeatMap
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split



# Read the JSON data from 'test.json'
with open('test.json') as file:
    data = json.load(file)
if False:
    with open('sold_test.json') as file:
        data_sold = json.load(file)
    
    
    
df = pd.DataFrame(data)

print(df.columns)
print(df.info())
print(df.isnull().sum())
df = df.drop(columns=['postcode','full_postcode','date_added'])

le = LabelEncoder()

# Fit and transform the 'type' column
df['type_encoded'] = le.fit_transform(df['propertySubType'])
df['branch_encoded'] = le.fit_transform(df['branchDisplayName'])
print(df[['propertySubType', 'type_encoded']])
pd.set_option('display.max_colwidth', None)
df.head(1).to_csv('viewing.csv')


df_x = df[['bedrooms','bathrooms','numberOfImages','numberOfFloorplans','numberOfVirtualTours','latitude','longitude','type_encoded']]
df_y = df['price_json']

X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=42)

