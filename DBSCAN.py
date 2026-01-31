from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


df = pd.read_csv(r'.\Clustering\kddcup.data_10_percent_corrected', header=None, sep=",")

feature_names = [
    "duration",  # Length of the connection in seconds
    "protocol_type",  # Type of protocol (e.g., TCP, UDP, ICMP)
    "service",  # Network service used (e.g., HTTP, FTP, SMTP)
    "flag",  # Status flag of the connection (e.g., SF for successful connection)
    "src_bytes",  # Number of data bytes sent from the source to the destination
    "dst_bytes",  # Number of data bytes sent from the destination to the source
    "land",  # Whether the connection is from/to the same host/port (1 for yes, 0 for no)
    "wrong_fragment",  # Number of "wrong" fragments in the connection
    "urgent",  # Number of urgent packets in the connection
    "hot",  # Number of "hot" indicators (e.g., accessing sensitive files)
    "num_failed_logins",  # Number of failed login attempts
    "logged_in",  # Whether the user is logged in (1 for yes, 0 for no)
    "num_compromised",  # Number of compromised conditions
    "root_shell",  # Whether a root shell was obtained (1 for yes, 0 for no)
    "su_attempted",  # Whether a `su root` command was attempted
    "num_root",  # Number of root accesses
    "num_file_creations",  # Number of file creation operations
    "num_shells",  # Number of shell prompts
    "num_access_files",  # Number of operations on access control files
    "num_outbound_cmds",  # Number of outbound commands in an FTP session
    "is_host_login",  # Whether the login is to the host (1 for yes, 0 for no)
    "is_guest_login",  # Whether the login is a guest login (1 for yes, 0 for no)
    "count",  # Number of connections to the same host in the past 2 seconds
    "srv_count",  # Number of connections to the same service in the past 2 seconds
    "serror_rate",  # Percentage of connections with SYN errors
    "srv_serror_rate",  # Percentage of connections to the same service with SYN errors
    "rerror_rate",  # Percentage of connections with REJ errors
    "srv_rerror_rate",  # Percentage of connections to the same service with REJ errors
    "same_srv_rate",  # Percentage of connections to the same service
    "diff_srv_rate",  # Percentage of connections to different services
    "srv_diff_host_rate",  # Percentage of connections to different hosts for the same service
    "dst_host_count",  # Number of connections to the same destination host
    "dst_host_srv_count",  # Number of connections to the same service on the destination host
    "dst_host_same_srv_rate",  # Percentage of connections to the same service on the destination host
    "dst_host_diff_srv_rate",  # Percentage of connections to different services on the destination host
    "dst_host_same_src_port_rate",  # Percentage of connections from the same source port to the destination host
    "dst_host_srv_diff_host_rate",
    # Percentage of connections to different hosts for the same service on the destination host
    "dst_host_serror_rate",  # Percentage of connections with SYN errors to the destination host
    "dst_host_srv_serror_rate",  # Percentage of connections with SYN errors to the same service on the destination host
    "dst_host_rerror_rate",  # Percentage of connections with REJ errors to the destination host
    "dst_host_srv_rerror_rate",  # Percentage of connections with REJ errors to the same service on the destination host
    "class"
]

df.columns = feature_names

# Identify non-numeric columns
non_numeric_cols = df.select_dtypes(exclude=['int64', 'float64']).columns
print("Non-numeric columns:", non_numeric_cols)

# Percentage of missing values
print(pd.DataFrame({'percent_missing': df.isnull().sum() * 100 / len(df)}))

# Calculate the class distribution of the target label
class_dist = df['class'].value_counts(normalize=True)

# Let's assume we want to sample 10% of the dataset
sample_fraction = 0.1

# Initialize an empty DataFrame to store the sampled data
df_sampled = pd.DataFrame()

# Sample without replacement for each class based on the class distribution
for class_label, class_percentage in class_dist.items():
    # Determine how many samples to take for each class based on the overall fraction and class distribution
    num_samples = int(np.floor(sample_fraction * len(df) * class_percentage))

    # Get the indices for the current class
    class_indices = df[df['class'] == class_label].index

    # Randomly sample from the class without replacement
    sampled_class = df.loc[class_indices].sample(n=num_samples, random_state=42, replace=False)

    # Drop the target column (class) from the sampled data
    sampled_class_without_target = sampled_class.drop(columns=['class'])

    # Append the sampled class to the final sampled dataset
    df_sampled = pd.concat([df_sampled, sampled_class_without_target])

# class_dist_sample = df_sampled['class'].value_counts(normalize=True)


# Verify the class distribution in the sampled data (without target)
print("Sampled data (without class column):")
print(df_sampled.head())

# Identify categorical columns
categorical_col = [col for col in df_sampled.columns if df_sampled[col].dtype == 'object']
print("Categorical columns:", categorical_col)

# Define a threshold percentage for grouping rare values
threshold = 2  # If a category appears in less than % of the data, we group it as "Other"

# Print category percentages for categorical columns
for col in categorical_col:
    category_percentages = df_sampled[col].value_counts(normalize=True) * 100
    print(f"Category percentages for {col}:\n{category_percentages}\n")
    rare_categories = category_percentages[category_percentages < threshold].index.tolist()  # Get rare categories
    # Replace rare categories with "Other"
    df_sampled[col] = df_sampled[col].replace(rare_categories, "Other")

for col in categorical_col:
    print(f"Updated category counts for {col}:")
    print(df_sampled[col].value_counts())
    print()  # For better readability

# One-hot encode categorical columns
data_encoded = pd.get_dummies(df_sampled, columns=categorical_col, drop_first=False)
# print(data_encoded.columns)

# Scale numeric columns
scaler = StandardScaler()
numeric_cols = data_encoded.select_dtypes(include=['float64', 'int64']).columns
data_encoded[numeric_cols] = scaler.fit_transform(data_encoded[numeric_cols])

# Apply PCA for dimensionality reduction
pca = PCA(n_components=17)  # Reduce to 10 principal components
df_pca = pca.fit_transform(data_encoded)

X = df_pca

explained_variance = pca.explained_variance_ratio_

# Print variance percentage of each principal component
for i, var in enumerate(explained_variance):
    print(f"Principal Component {i + 1}: {var * 100:.2f}% variance explained")

# (Optional) Check cumulative explained variance
cumulative_variance = np.cumsum(explained_variance)
print("\nCumulative Explained Variance:")
for i, cum_var in enumerate(cumulative_variance):
    print(f"Up to Principal Component {i + 1}: {cum_var * 100:.2f}% variance explained")

feature_names = data_encoded.columns  # Get original feature names

# Create a DataFrame showing how each original feature contributes to each principal component
pca_components_df = pd.DataFrame(pca.components_, columns=feature_names,
                                 index=[f"PC{i + 1}" for i in range(len(pca.components_))])

# Display the component contributions
print(pca_components_df)

# Define DBSCAN parameters (you can tune these manually)
eps_value = 0.6  # Set your desired value
min_samples_value = 14  # Set your desired value

# Apply DBSCAN to PCA-reduced data
dbscan = DBSCAN(eps=eps_value, min_samples=min_samples_value)
dbscan_labels = dbscan.fit_predict(X)

# Filter out noise points (-1 label) before computing silhouette score
mask = dbscan_labels != -1
if len(set(dbscan_labels[mask])) > 1:
    score = silhouette_score(X[mask], dbscan_labels[mask])
    print(f"Silhouette Score (excluding noise): {score:.4f}")
else:
    print("Silhouette Score cannot be computed: less than 2 clusters formed (excluding noise).")

# Optional: Check how many points were labeled as noise
n_noise = np.sum(dbscan_labels == -1)
print(f"Number of noise points: {n_noise}")
print(f"Number of clusters (excluding noise): {len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)}")

tsne = TSNE(n_components=2, perplexity=23, random_state=42, learning_rate=200)

#X_tsne = tsne.fit_transform(X)
X_tsne = np.load(r'.\Clustering\x_tsne.npy')

plt.figure(figsize=(8, 6))

# Unique cluster labels excluding outliers
unique_labels = sorted(set(dbscan_labels))
unique_clusters = [label for label in unique_labels if label != -1]

# Use the updated approach: Access colormap directly from pyplot
cmap = plt.get_cmap('tab20')  # Access colormap from pyplot
colors = [cmap(i / len(unique_clusters)) for i in range(len(unique_clusters))]  # Normalize color indices

# Plot clusters
for idx, label in enumerate(unique_clusters):
    mask = dbscan_labels == label
    plt.scatter(
        X_tsne[mask, 0],
        X_tsne[mask, 1],
        c=[colors[idx]],
        s=5,
        alpha=0.5,
        label=f"Cluster {label}"
    )

# Plot outliers in red
outliers_mask = dbscan_labels == -1
plt.scatter(
    X_tsne[outliers_mask, 0],
    X_tsne[outliers_mask, 1],
    c='black',
    label='Outliers',
    s=10,
    alpha=0.6
)

plt.legend()
plt.title("t-SNE with DBSCAN Clusters (Black = outliers)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.savefig("tsne_dbscan.png")
plt.show()
plt.close()

