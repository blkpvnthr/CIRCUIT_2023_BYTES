import pandas as pd
import sklearn.cluster as clusters
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
# Load the data from the CSV files
deaths_data = pd.read_csv('deaths.csv')

deaths_data['Deaths_Percent'] = (deaths_data['Deaths'] / deaths_data['Population']) * 100
print(deaths_data['Deaths_Percent'])

from sklearn.cluster import KMeans

# Number of clusters you want to create
num_clusters = 5  # You can adjust this value based on your requirements

# Selecting the columns for clustering
X = deaths_data[['Latitude', 'Longitude', 'Deaths_Percent']]

# Initializing k-means with the specified number of clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=42)

# Fit the k-means model to the data
kmeans.fit(X)

# Add the cluster labels to the DataFrame
deaths_data['Cluster'] = kmeans.labels_

# Let's see the clustered data
print(deaths_data.head())
