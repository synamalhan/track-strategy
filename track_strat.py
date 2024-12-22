# Import FastF1 if not already done
import fastf1
import pandas as pd 
# Enable caching and load session as before
fastf1.Cache.enable_cache('cache')  
session = fastf1.get_session(2023, 'Bahrain', 'Race')
session.load()

# Extract all lap data
laps = session.laps  # Gets laps for all drivers
print(laps.head())  # Preview the first few rows of lap data

# Pick a specific driver (e.g., Max Verstappen, driver code 'VER')
driver_laps = laps.pick_driver('VER')  # Replace 'VER' with another driver's code if needed
print(driver_laps.head())  # Preview the laps for the selected driver

# Find the driver's fastest lap
fastest_lap = driver_laps.pick_fastest()
print(f"Fastest Lap Time: {fastest_lap['LapTime']}")

# Extract telemetry data for the fastest lap
telemetry = fastest_lap.get_telemetry()

# Display the first few rows of telemetry data
print(telemetry.head())

# Plot Speed vs. Distance
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(telemetry['Distance'], telemetry['Speed'], label='Speed (km/h)')
plt.xlabel('Distance (m)')
plt.ylabel('Speed (km/h)')
plt.title('Speed Profile of Fastest Lap')
plt.legend()
plt.show()

# Plot Throttle and Brake vs. Distance
plt.figure(figsize=(10, 5))
plt.plot(telemetry['Distance'], telemetry['Throttle'], label='Throttle (%)', color='green')
plt.plot(telemetry['Distance'], telemetry['Brake'], label='Brake Intensity', color='red')
plt.xlabel('Distance (m)')
plt.ylabel('Intensity')
plt.title('Throttle and Brake Intensity Over Distance')
plt.legend()
plt.show()


import matplotlib.pyplot as plt

# Extract positional data from telemetry
x = telemetry['X']  # X-coordinate
y = telemetry['Y']  # Y-coordinate
brake = telemetry['Brake']  # Braking intensity

# Plot the track layout
plt.figure(figsize=(10, 10))
scatter = plt.scatter(x, y, c=brake, cmap='Reds', s=5)  # Color by braking intensity
plt.colorbar(scatter, label='Brake Intensity')
plt.title('Track Map with Braking Zones')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.show()

# Extract positional data and speed from telemetry
x = telemetry['X']
y = telemetry['Y']
speed = telemetry['Speed']  # Speed at each point

# Plot the track layout with speed as a color map
plt.figure(figsize=(10, 10))
scatter = plt.scatter(x, y, c=speed, cmap='viridis', s=5)  # Viridis is a good colormap for speed
plt.colorbar(scatter, label='Speed (km/h)')
plt.title('Track Map with Speed Zones')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.show()

# Extract positional data, speed, and braking from telemetry
x = telemetry['X']
y = telemetry['Y']
speed = telemetry['Speed']
brake = telemetry['Brake']

# Plot the track layout with combined metrics
plt.figure(figsize=(10, 10))
scatter = plt.scatter(x, y, c=speed, cmap='viridis', s=brake * 50, alpha=0.7)  # s controls point size; alpha for transparency
plt.colorbar(scatter, label='Speed (km/h)')
plt.title('Track Map with Speed and Braking Zones')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.show()



#------------Clustering---------------------

from sklearn.cluster import KMeans
import numpy as np

# Prepare data for clustering
# Use Speed, Brake, and Distance to create a feature set
features = telemetry[['Speed', 'Brake', 'Distance']].fillna(0)  # Handle any missing values with 0

# Apply K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42)  # 3 clusters (e.g., straights, corners, braking zones)
clusters = kmeans.fit_predict(features)

# Add cluster labels to telemetry data
telemetry['Cluster'] = clusters

# Plot the track with clusters
plt.figure(figsize=(10, 10))
scatter = plt.scatter(
    telemetry['X'], telemetry['Y'], c=telemetry['Cluster'], cmap='Set1', s=10, alpha=0.7
)
plt.colorbar(scatter, label='Cluster')
plt.title('Track Segmentation with Clusters')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.show()


# #------------n-values----------------------

# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt

# # Test multiple cluster numbers
# inertia = []
# cluster_range = range(1, 10)  # Try clusters from 1 to 10
# for k in cluster_range:
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     kmeans.fit(features)
#     inertia.append(kmeans.inertia_)

# # Plot inertia vs. number of clusters
# plt.figure(figsize=(8, 5))
# plt.plot(cluster_range, inertia, marker='o')
# plt.title('Elbow Method for Optimal Clusters')
# plt.xlabel('Number of Clusters')
# plt.ylabel('Inertia')
# plt.show()


# from sklearn.metrics import silhouette_score

# silhouette_scores = []
# for k in range(2, 10):  # Start from 2 clusters (Silhouette requires at least 2 clusters)
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     kmeans.fit(features)
#     score = silhouette_score(features, kmeans.labels_)
#     silhouette_scores.append(score)

# # Plot silhouette scores
# plt.figure(figsize=(8, 5))
# plt.plot(range(2, 10), silhouette_scores, marker='o')
# plt.title('Silhouette Score for Optimal Clusters')
# plt.xlabel('Number of Clusters')
# plt.ylabel('Silhouette Score')
# plt.show()

# Analyze clusters: Compute summary statistics for each cluster
cluster_stats = telemetry.groupby('Cluster').agg({
    'Speed': ['mean', 'max'],
    'Brake': ['mean', 'max'],
    'Throttle': ['mean', 'max'],
    'Distance': ['count']  # Number of data points in each cluster
}).reset_index()

# Flatten column names for readability
cluster_stats.columns = ['Cluster', 'Mean Speed', 'Max Speed', 'Mean Brake', 'Max Brake', 
                         'Mean Throttle', 'Max Throttle', 'Data Points']

# Print cluster statistics
print("Cluster Summary Statistics:")
print(cluster_stats)

# Visualize speed and braking intensity across clusters
plt.figure(figsize=(12, 6))

# Plot average speed by cluster
plt.subplot(1, 2, 1)
plt.bar(cluster_stats['Cluster'], cluster_stats['Mean Speed'], color='skyblue')
plt.title('Average Speed by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Speed (km/h)')

# Plot average braking intensity by cluster
plt.subplot(1, 2, 2)
plt.bar(cluster_stats['Cluster'], cluster_stats['Mean Brake'], color='salmon')
plt.title('Average Braking Intensity by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Braking Intensity')

plt.tight_layout()
plt.show()

# Convert 'Brake' to numeric if it is boolean
telemetry['Brake'] = telemetry['Brake'].astype(int)

# Identifying overtaking zones with speed and braking
overtaking_zones = telemetry[(telemetry['Speed'] > telemetry['Speed'].quantile(0.75)) & 
                             (telemetry['Brake'] > telemetry['Brake'].quantile(0.75))]

# Plotting the track with overtaking zones
plt.figure(figsize=(10, 10))
plt.scatter(telemetry['X'], telemetry['Y'], c=telemetry['Cluster'], cmap='Set1', s=10, alpha=0.7)  # Base track plot

# Highlight overtaking zones
plt.scatter(overtaking_zones['X'], overtaking_zones['Y'], c='red', s=30, label='Overtaking Zones')

plt.title('Track with Overtaking Zones Highlighted')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.show()


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Prepare the data
features = telemetry[['Cluster', 'Speed', 'Brake', 'Throttle', 'Distance']]
target = (telemetry['Cluster'].isin(overtaking_zones['Cluster'])).astype(int)  # Binary target (0 or 1)

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))



# Select features for clustering (using all the telemetry data, not just the fastest lap)
features = ['Speed', 'Throttle', 'Brake', 'RPM', 'nGear']  # You can adjust the features as needed
X = telemetry[features]  # Use the entire telemetry dataset

# Scale the data (important for clustering)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply KMeans clustering on the full dataset
from sklearn.cluster import KMeans
n_clusters = 5  # Set the number of clusters (5 clusters as an example)
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
telemetry['Cluster'] = kmeans.fit_predict(X_scaled)

# Display the clustered data
print(telemetry[['Time', 'Speed', 'Throttle', 'Brake', 'RPM', 'nGear', 'Cluster']].head())


# Summarize cluster statistics
cluster_summary = telemetry.groupby('Cluster')[['Speed', 'Throttle', 'Brake', 'RPM', 'nGear']].agg(['mean', 'max', 'min', 'std'])
print(cluster_summary)

import matplotlib.pyplot as plt
import seaborn as sns

# 2D visualization: You can use two features (e.g., Speed and Throttle) to visualize clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=telemetry['Speed'], y=telemetry['Throttle'], hue=telemetry['Cluster'], palette='viridis')
plt.title('Cluster Visualization (Speed vs Throttle)')
plt.xlabel('Speed')
plt.ylabel('Throttle')
plt.legend(title='Cluster')
plt.show()

# Alternatively, for 3D visualization (if you have 3 features), you can use matplotlib 3D plotting
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(telemetry['Speed'], telemetry['Throttle'], telemetry['RPM'], c=telemetry['Cluster'], cmap='viridis')
ax.set_xlabel('Speed')
ax.set_ylabel('Throttle')
ax.set_zlabel('RPM')
plt.title('3D Cluster Visualization (Speed, Throttle, RPM)')
plt.show()

# For example, let's see if clusters are related to lap times (if you have lap time data)
# Assuming 'Time' or a similar metric represents lap time
telemetry['Time'] = pd.to_timedelta(telemetry['Time'])  # Convert 'Time' to timedelta if necessary

cluster_time_summary = telemetry.groupby('Cluster')['Time'].agg(['mean', 'min', 'max'])
print(cluster_time_summary)

# Export the clustered telemetry data to a CSV file
telemetry.to_csv('clustered_telemetry_data.csv', index=False)

# Identify clusters with higher speed and lower throttle or brake usage
threshold_speed = 250  # Adjust this based on your data
threshold_brake = 0.1  # Adjust based on your data (lower values for braking)

# Filter data for overtaking
overtaking_clusters = telemetry[(telemetry['Speed'] > threshold_speed) & (telemetry['Brake'] < threshold_brake)]

# Step 2: If no overtaking data, handle that case
if overtaking_clusters.empty:
    print("No overtaking data found with the given thresholds.")
else:
    # Step 3: Prepare data for clustering (features: X, Y, Speed)
    X = overtaking_clusters[['X', 'Y', 'Speed']]

    # Step 4: Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 5: Apply KMeans Clustering (adjust n_clusters as necessary)
    kmeans = KMeans(n_clusters=5, random_state=42)
    overtaking_clusters['Cluster'] = kmeans.fit_predict(X_scaled)

    # Step 6: Visualize the clusters on the track using X and Y coordinates
    plt.figure(figsize=(10, 6))
    
    # Scatter plot with clusters colored
    sns.scatterplot(x=overtaking_clusters['X'], y=overtaking_clusters['Y'], hue=overtaking_clusters['Cluster'], palette='coolwarm', s=100, marker='o')
    plt.title('Overtaking Zones (Clusters with Higher Speed and Lower Brake)')
    plt.xlabel('Track X')
    plt.ylabel('Track Y')
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

# Optional: Output the cluster centers to understand where overtakes are happening
print("Cluster Centers (X, Y, Speed):")
print(kmeans.cluster_centers_)



# Identify clusters with high braking
braking_clusters = telemetry[telemetry['Brake'] > telemetry['Brake'].quantile(0.75)]

# Visualize braking clusters
sns.scatterplot(x=braking_clusters['X'], y=braking_clusters['Y'], hue=braking_clusters['Cluster'], palette='coolwarm')
plt.title('Optimal Braking Points (Clusters with High Brake)')
plt.xlabel('Track X')
plt.ylabel('Track Y')
plt.legend(title='Cluster')
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Assuming 'Time' is the lap time or sector time
X = telemetry[['Speed', 'Throttle', 'Brake', 'RPM', 'nGear']]  # Features
y = telemetry['Time'].dt.total_seconds()  # Convert time to total seconds for prediction

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a random forest regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, predictions)
print(f'Mean Absolute Error: {mae}')






# from sklearn.model_selection import RandomizedSearchCV

# param_distributions = {
#     'n_estimators': [100, 200, 500],
#     'max_depth': [5, 10, 20],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
# }

# random_search = RandomizedSearchCV(
#     estimator=RandomForestRegressor(random_state=42),
#     param_distributions=param_distributions,
#     n_iter=50,  # Number of random combinations to try
#     scoring='neg_mean_absolute_error',
#     cv=5,
#     random_state=42,
#     n_jobs=-1
# )
# random_search.fit(X_train, y_train)

# print("Best Parameters:", random_search.best_params_)


from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

# Define the best parameters
best_params = {
    'n_estimators': 500,
    'min_samples_split': 10,
    'min_samples_leaf': 1,
    'max_depth': 10,
}

# Instantiate the model with the best parameters
xgb_model = XGBRegressor(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    learning_rate=0.005,  
    min_child_weight=best_params['min_samples_split'],
    reg_alpha=0,  # L1 regularization term, can help with overfitting
    reg_lambda=10,  # L2 regularization term
    random_state=42
)

# Fit the model
xgb_model.fit(
    X_train, y_train
)


# Predict and evaluate
y_pred = xgb_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

print("Optimized MAE:", mae)

from xgboost import plot_importance
import matplotlib.pyplot as plt

plot_importance(xgb_model)
plt.show()


