# =============================================================================
# PCA Analysis for 80 Cereals Dataset
# =============================================================================

# %% Cell 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %% Cell 2: Download Dataset
import kagglehub
import os

# Download latest version
path = kagglehub.dataset_download("crawford/80-cereals")
print("Path to dataset files:", path)

# %% Cell 3: Load and Preview Data
# Construct the full path to the cereal CSV file
csv_filename = 'cereal.csv'
full_csv_path = os.path.join(path, csv_filename)

df = pd.read_csv(full_csv_path)
display(df.head())

# %% Cell 4: Rename Columns (English to French)
column_renaming_map = {
    'name': 'Nom',
    'mfr': 'Fabricant',
    'type': 'Type',
    'calories': 'Calories',
    'protein': 'Proteines',
    'fat': 'Graisses',
    'sodium': 'Sodium',
    'fiber': 'Fibres',
    'carbo': 'Glucides',
    'sugars': 'Sucres',
    'potass': 'Potassium',
    'vitamins': 'Vitamines',
    'shelf': 'Etagere',
    'weight': 'Poids',
    'cups': 'Tasses',
    'rating': 'Note'
}

df = df.rename(columns=column_renaming_map)
display(df.head())

# %% Cell 5: Set Index
# Set the cereal name as the index
df = df.set_index('Nom')

# %% Cell 6: Data Shape
print(f"Dataset Shape: {df.shape}")

# %% Cell 7: Display Head
display(df.head())

# %% Cell 8: Missing Values
# Show count of missing values for each column
print("Missing values per column:")
print(df.isnull().sum())

# %% Cell 9: Data Info
print("\nData Types:")
print(df.dtypes)

# %% Cell 10: Select Numeric Features for PCA
from sklearn.preprocessing import StandardScaler

# Select only numeric columns for PCA
# Exclude non-numeric columns (Fabricant, Type) and the target variable (Note)
numeric_cols = ['Calories', 'Proteines', 'Graisses', 'Sodium', 'Fibres', 
                'Glucides', 'Sucres', 'Potassium', 'Vitamines', 'Etagere', 
                'Poids', 'Tasses']
numeric_features = df[numeric_cols]

print("Features to scale:")
print(numeric_features.columns.tolist())

# %% Cell 11: Check Missing Values in Numeric Features
# Check if you have missing data
missing_count = numeric_features.isnull().sum().sum()
print(f"Total missing values in numeric features: {missing_count}")

# If you have missing values, show which columns
if missing_count > 0:
    print("\nColumns with missing values:")
    print(numeric_features.isnull().sum()[numeric_features.isnull().sum() > 0])

# %% Cell 12: Handle Missing Values (Imputation)
from sklearn.impute import SimpleImputer

# Impute missing values with median
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(numeric_features)

# %% Cell 13: Scale the Data
# Create scaler and fit + transform
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Convert back to DataFrame
X_preprocessed = pd.DataFrame(
    X_scaled, 
    columns=numeric_features.columns,
    index=df.index
)

print("✓ Data scaled successfully!")
print(f"Shape: {X_preprocessed.shape}")
display(X_preprocessed.head())

# %% Cell 14: Verify Scaling
# Check that mean is ~0 and std is ~1
print("\nMean (should be close to 0):")
print(X_preprocessed.mean().round(10))

print("\nStd (should be close to 1):")
print(X_preprocessed.std().round(3))

# Show before and after with rounded values
print("="*80)
print("COMPARISON: Before vs After Scaling (First Cereal)")
print("="*80)

comparison_df = pd.DataFrame({
    'Feature': numeric_features.columns,
    'Original Value': numeric_features.iloc[0].values,
    'Scaled Value': X_preprocessed.iloc[0].values.round(3)
})

print(comparison_df.to_string(index=False))

# %% Cell 15: Run 2D PCA
from sklearn.decomposition import PCA

# Fit PCA with 2 components for 2D visualization
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_preprocessed)

print("="*60)
print("2D PCA RESULTS")
print("="*60)
print(f"Original features: {X_preprocessed.shape[1]}")
print(f"Reduced to: {X_pca_2d.shape[1]} components")
print(f"Variance explained by PC1: {pca_2d.explained_variance_ratio_[0]:.2%}")
print(f"Variance explained by PC2: {pca_2d.explained_variance_ratio_[1]:.2%}")
print(f"Total variance explained: {pca_2d.explained_variance_ratio_.sum():.2%}")

# Visualize 2D PCA - color by Manufacturer
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], 
                      c=df['Fabricant'].astype('category').cat.codes, 
                      cmap='viridis', alpha=0.7, s=100)
plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%} variance)')
plt.title('2D PCA Visualization of Cereal Data (Colored by Manufacturer)')
plt.colorbar(scatter, label='Manufacturer Code')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% Cell 16: Run 3D PCA
# Fit PCA with 3 components for 3D visualization
pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X_preprocessed)

print("="*60)
print("3D PCA RESULTS")
print("="*60)
print(f"Original features: {X_preprocessed.shape[1]}")
print(f"Reduced to: {X_pca_3d.shape[1]} components")
print(f"Variance explained by PC1: {pca_3d.explained_variance_ratio_[0]:.2%}")
print(f"Variance explained by PC2: {pca_3d.explained_variance_ratio_[1]:.2%}")
print(f"Variance explained by PC3: {pca_3d.explained_variance_ratio_[2]:.2%}")
print(f"Total variance explained: {pca_3d.explained_variance_ratio_.sum():.2%}")

# 3D Visualization
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2],
                     c=df['Fabricant'].astype('category').cat.codes,
                     cmap='viridis', alpha=0.7, s=100)
ax.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]:.1%})')
ax.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]:.1%})')
ax.set_title('3D PCA Visualization of Cereal Data')
plt.tight_layout()
plt.show()

# %% Cell 17: Scree Plot
# Fit PCA with all components
pca_full = PCA()
pca_full.fit(X_preprocessed)

# Scree plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.bar(range(1, len(pca_full.explained_variance_ratio_) + 1), 
        pca_full.explained_variance_ratio_, alpha=0.7, label='Individual')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained Ratio')
plt.title('Scree Plot')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(range(1, len(pca_full.explained_variance_ratio_) + 1), 
         np.cumsum(pca_full.explained_variance_ratio_), 'bo-')
plt.axhline(y=0.9, color='r', linestyle='--', label='90% threshold')
plt.axhline(y=0.95, color='g', linestyle='--', label='95% threshold')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Variance Explained')
plt.title('Cumulative Variance Explained')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print how many components needed for 90% and 95% variance
cumsum = np.cumsum(pca_full.explained_variance_ratio_)
n_90 = np.argmax(cumsum >= 0.90) + 1
n_95 = np.argmax(cumsum >= 0.95) + 1
print(f"\nComponents needed to explain 90% variance: {n_90}")
print(f"Components needed to explain 95% variance: {n_95}")

# %% Cell 18: Feature Loadings (Biplot)
# Get loadings
loadings = pca_2d.components_.T

# Create a biplot
plt.figure(figsize=(12, 10))
plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], alpha=0.5, s=50)

# Plot feature vectors
scale = 5  # Scale factor for arrows
for i, feature in enumerate(numeric_features.columns):
    plt.arrow(0, 0, loadings[i, 0] * scale, loadings[i, 1] * scale,
              head_width=0.15, head_length=0.1, fc='red', ec='red', alpha=0.8)
    plt.text(loadings[i, 0] * scale * 1.15, loadings[i, 1] * scale * 1.15,
             feature, fontsize=10, ha='center', va='center')

plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%} variance)')
plt.title('Biplot: Cereal Data with Feature Loadings')
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
plt.tight_layout()
plt.show()

# %% Cell 19: Loading Interpretation
print("="*60)
print("FEATURE LOADINGS INTERPRETATION")
print("="*60)

# Create a loadings dataframe
loadings_df = pd.DataFrame(
    pca_full.components_[:3].T,
    columns=['PC1', 'PC2', 'PC3'],
    index=numeric_features.columns
)

print("\nTop 3 Principal Components Loadings:")
print(loadings_df.round(3))

print("\n" + "="*60)
print("Features contributing most to each PC:")
print("="*60)

for pc in ['PC1', 'PC2', 'PC3']:
    print(f"\n{pc}:")
    sorted_loadings = loadings_df[pc].abs().sort_values(ascending=False)
    for feature in sorted_loadings.head(3).index:
        value = loadings_df.loc[feature, pc]
        direction = "+" if value > 0 else "-"
        print(f"  {feature}: {value:.3f} ({direction})")

# %% Cell 20: Clustering with K-Means on PCA Components
from sklearn.cluster import KMeans

# Use first 2 PCs for clustering
X_for_clustering = X_pca_2d

# Find optimal k using elbow method
inertias = []
K_range = range(2, 10)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_for_clustering)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid(True, alpha=0.3)

# Let's use k=4 clusters
k = 4
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_for_clustering)

plt.subplot(1, 2, 2)
scatter = plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], 
                      c=df['Cluster'], cmap='Set1', alpha=0.7, s=100)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c='black', marker='X', s=200, label='Centroids')
plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%} variance)')
plt.title(f'K-Means Clustering (k={k}) on PCA Components')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% Cell 21: Cluster Analysis
print("="*60)
print("CLUSTER ANALYSIS")
print("="*60)

for cluster in range(k):
    cluster_cereals = df[df['Cluster'] == cluster]
    print(f"\nCluster {cluster} ({len(cluster_cereals)} cereals):")
    print("-" * 40)
    
    # Show some sample cereals in this cluster
    print("Sample cereals:", list(cluster_cereals.index[:5]))
    
    # Show average characteristics
    cluster_features = cluster_cereals[numeric_cols].mean()
    print(f"\nAverage characteristics:")
    print(cluster_features.round(2))

# %% Cell 22: Summary Statistics by Cluster (Heatmap)
# Calculate mean of numeric features by cluster
cluster_means = df.groupby('Cluster')[numeric_cols].mean()

plt.figure(figsize=(12, 6))
sns.heatmap(cluster_means.T, annot=True, fmt='.1f', cmap='viridis')
plt.title('Average Feature Values by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

print("\n✓ PCA Analysis Complete!")
