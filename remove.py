import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity


if __name__ == "__main__":

    data = pd.read_csv("./data/transformed_data_multiple_features.csv")
    X = data[["GDP per capita", "Social support", "Healthy life expectancy"]]

    num_clusters = 5
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(X)
    cluster_labels = kmeans.labels_
    data['Cluster'] = cluster_labels
    cluster_centers = kmeans.cluster_centers_

    cluster_counts = data['Cluster'].value_counts().sort_index()
    print("Number of data points in each cluster:")
    print(cluster_counts)

    cluster_stats = data.groupby('Cluster').mean().reset_index()
    print("Cluster Statistics:")
    print(cluster_stats)

    features = ['rent', 'BEDROOMS', 'BATHROOMS']
    Y = cluster_stats[features].values


    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    Y_scaled = scaler.fit_transform(Y)

    num_clusters = 5
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    print("Silhouette Score:", silhouette_avg)
    ch_score = calinski_harabasz_score(X_scaled, cluster_labels)
    print("Calinski-Harabasz Index:", ch_score)
    db_score = davies_bouldin_score(X_scaled, cluster_labels)
    print("Davies-Bouldin Index:", db_score)

    dissimilarity_matrix = euclidean_distances(Y_scaled)
    similarity_matrix = cosine_similarity(Y_scaled)

    print("Dissimilarity matrix \n", dissimilarity_matrix)
    print("Similarity matrix \n", similarity_matrix)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X['BEDROOMS_johnson'], X['BATHROOMS_johnson'], X['rent_johnson'], c=cluster_labels, cmap='rainbow')
    ax.set_xlabel('rent_johnson')
    ax.set_ylabel('BEDROOMS_johnson')
    ax.set_zlabel('BATHROOMS_johnson')
    ax.set_title('K means Clustering (3D)')
    plt.show()

