import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import pandas as pd

# Step 1: Simulated User-Item Ratings Matrix (Sparse Matrix)
# Rows: Users, Columns: Items
np.random.seed(42)  # For reproducibility
num_users = 1000
num_items = 500

# Simulating sparse ratings matrix (most entries are zero)
ratings_matrix = np.random.randint(0, 6, size=(num_users, num_items))
ratings_matrix[ratings_matrix < 4] = 0  # Making it sparse

print(ratings_matrix)
# Step 2: Dimensionality Reduction with PCA
# Reduce the dimensionality of the sparse ratings matrix
latent_dim = 300  # Number of latent factors
pca = PCA(n_components=latent_dim)
user_features = pca.fit_transform(ratings_matrix)

print(f"Original dimensions: {ratings_matrix.shape}")
print(f"Reduced dimensions: {user_features.shape}")

# Step 3: Fit Gaussian Mixture Model (GMM)
k_clusters = 5  # Number of clusters

# GMM with diagonal covariance for simplicity
gmm = GaussianMixture(n_components=k_clusters, covariance_type='diag', random_state=42)
gmm.fit(user_features)

# Step 4: Assign Users to Clusters
user_clusters = gmm.predict(user_features)

# Calculate silhouette score to evaluate clustering
sil_score = silhouette_score(user_features, user_clusters)
print(f"Silhouette Score: {sil_score:.2f}")

# Step 5: Recommendations Based on Clusters
# Example: Recommend top items from the cluster centroid
cluster_centers = gmm.means_  # Cluster centers in the latent space


# Function to get recommended items for a user
def recommend_items(user_id, top_n=5):
    cluster_id = user_clusters[user_id]
    # Find the top-rated items in this cluster (simple heuristic)
    cluster_ratings = ratings_matrix[user_clusters == cluster_id].mean(axis=0)
    recommended_items = np.argsort(-cluster_ratings)[:top_n]
    return recommended_items


print(user_features)
# Example Recommendation for User 0
user_id = 1
recommendations = recommend_items(user_id)
print(f"Recommended items for User {user_id}: {recommendations}")
