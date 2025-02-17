import json
import pandas as pd
import folium
from folium.plugins import HeatMap
import seaborn as sns
import matplotlib.pyplot as plt

# Read the JSON data from 'test.json'
with open('test.json') as file:
    data = json.load(file)
with open('sold_test.json') as file:
    data_sold = json.load(file)
df_sold = pd.DataFrame(data_sold)
print(df_sold.shape)
# Create a DataFrame from the JSON data
df = pd.DataFrame(data)
df_sold = df_sold.rename(columns={'last_price': 'price', 'lat': 'latitude', 'lng': 'longitude'})
df_price_region_sold = df_sold[['price','longitude','latitude']]
# Print the column names
print(df_sold.columns)
df_price_region = df[['price','longitude','latitude']]
df_combined = pd.concat([df_price_region, df_price_region_sold], ignore_index=True)

# Create a base map centered on the region
map_center = [df_combined['latitude'].mean(), df_combined['longitude'].mean()]
map_obj = folium.Map(location=map_center, zoom_start=15)
print(df_combined.shape)
# Create a color scale based on the price range
min_price = df_combined['price'].min()
max_price = df_combined['price'].max()
color_scale = lambda price: 'red' if price > (max_price - min_price) * 0.75 else 'orange' if price > (max_price - min_price) * 0.5 else 'green' if price > (max_price - min_price) * 0.25 else 'blue'

# Add markers for each house
for idx, row in df_combined.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=5,
        color=color_scale(row['price']),
        fill=True,
        fill_color=color_scale(row['price']),
        fill_opacity=0.7,
        popup=f"Price: {row['price']}"
    ).add_to(map_obj)

# Save the map as an HTML file
map_obj.save('scatter_plot.html')

# Group the data by latitude and longitude and calculate the average price for each location
location_avg_price = df_combined.groupby(['latitude', 'longitude'])['price'].mean().reset_index()

# Create a list of [latitude, longitude, average_price] for each location
heat_data = location_avg_price[['latitude', 'longitude', 'price']].values.tolist()

# Create a base map centered on the region
map_center = [df_combined['latitude'].mean(), df_combined['longitude'].mean()]
map_obj = folium.Map(location=map_center, zoom_start=15)

# Add the heatmap layer to the map
HeatMap(heat_data).add_to(map_obj)

# Save the map as an HTML file
map_obj.save('average_price_heatmap.html')

# Create a Seaborn graph of bedroom count vs price
plt.figure(figsize=(10, 6))
sns.barplot(x='bedrooms', y='price', data=df_sold)
plt.title('Bedroom Count vs Price')
plt.xlabel('Bedroom Count')
plt.ylabel('Price')
plt.show()
