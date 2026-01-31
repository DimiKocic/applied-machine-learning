from sklearn.metrics import precision_recall_fscore_support, f1_score, precision_score, recall_score, accuracy_score, \
    classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import shap
from scipy.io import arff
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import arff
import joblib
from sklearn.decomposition import PCA


# Commented for performance issues
# Read ARFF file
# with open("IMDB-F.arff", 'r') as f:
#     dataset = arff.load(f)
#
# # Convert to pandas DataFrame
# df = pd.DataFrame(dataset['data'], columns=[attr[0] for attr in dataset['attributes']])

df = pd.read_csv(r'./RandomForest/IMDB-F_CSV.csv', header=0, sep=',')

genre_columns = ['Drama', 'Comedy', 'Short', 'Documentary', 'Romance']
for col in genre_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Corrected filter condition
df = df[(df['Drama'] == 1) | (df['Comedy'] == 1) | (df['Short'] == 1) | (df['Documentary'] == 1) | (df['Romance'] == 1)]
df = df.reset_index(drop=True)

# Drop unused genre columns
df.drop(columns=['Action', 'Thriller', 'Crime', 'Family', 'Animation',
                 'Adventure', 'Horror', 'Sci-Fi', 'Fantasy', 'Mystery',
                 'Music', 'Western', 'War', 'Musical', 'History', 'Biography',
                 'Sport', 'Adult', 'Reality-TV', 'Game-Show', 'News',
                 'Talk-Show', 'Film-Noir'], inplace=True)


# Percentage of missing values
print(pd.DataFrame({'percent_missing': df.isnull().sum() * 100 / len(df)}))

# Extract multilabel targets
y = df.iloc[:, :5]  # First 5 columns: Drama, Comedy, Short, Documentary, Romance
y = y.astype(int)

# Extract features
X = df.iloc[:, 5:]

# Identify and drop highly dominant columns
highly_dominant_columns = [
    col for col in X.columns if (X[col].value_counts(normalize=True).max() >= 0.99)
]
print(f"Columns with one value dominating 99% or more: {highly_dominant_columns}, {len(highly_dominant_columns)}")
X.drop(columns=highly_dominant_columns, inplace=True)



##############################################################################################
# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features using StandardScaler (important for neural networks)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Apply PCA for dimensionality reduction
pca = PCA(n_components=0.90)  # Retain 95% of the variance
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Print the number of components after PCA
print(f"Number of components after PCA: {X_train_pca.shape[1]}")


# Create a base classifier (Decision Tree)
base_clf = DecisionTreeClassifier(max_depth=2)
#
# Use MultiOutputClassifier to handle multi-label classification
# AdaBoost with One-vs-Rest strategy
#clf = MultiOutputClassifier(AdaBoostClassifier(base_clf, n_estimators=50, learning_rate=0.001), n_jobs=-1)

# Commented for performance issues
# Fit and save the model
# clf.fit(X_train, y_train)
# joblib.dump(clf, 'clf_results.pkl')

#Load Saved File
clf1 = joblib.load(r'./RandomForest/clf1.pkl')

y_pred = clf1.predict(X_test)

# Classification report
print('Classification Report:')
print(classification_report(y_test, y_pred, target_names=y.columns,zero_division=1))

# Calculate per-label accuracy
label_accuracies = {}
for i, label in enumerate(y_test.columns):
    acc = accuracy_score(y_test.iloc[:, i], y_pred[:, i])
    label_accuracies[label] = acc

# Sort labels by accuracy (descending) and select top N
top_n = 5
top_labels = sorted(label_accuracies, key=label_accuracies.get, reverse=True)[:top_n]


# Compute probabilities for ROC curve plotting
# MultiOutputClassifier returns a list of arrays - we need to stack them
y_prob = np.array([pred[:, 1] for pred in clf1.predict_proba(X_test)]).T

# Plot ROC curves for each genre
plt.figure(figsize=(10, 8))
for label_name in top_labels:
    i = y_test.columns.get_loc(label_name)
    fpr, tpr, _ = roc_curve(y_test.iloc[:, i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{label_name} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"Multilabel ROC Curve - Top {top_n} Labels by Accuracy")
plt.legend(loc="lower right", bbox_to_anchor=(1.6, 0))
plt.tight_layout()
plt.savefig('multilabel_roc_curve_top_labels.png')
plt.show()

########################################################################################
#SHAP
#Load a saved file
clf = joblib.load(r'./RandomForest/clf.pkl')
y_pred = clf.predict(X_test)

# Calculate per-label accuracy
label_accuracies = {}
for i, label in enumerate(y_test.columns):
    acc = accuracy_score(y_test.iloc[:, i], y_pred[:, i])
    label_accuracies[label] = acc

# Sort labels by accuracy (descending) and select top N
top_n = 5
top_labels = sorted(label_accuracies, key=label_accuracies.get, reverse=True)[:top_n]
print(f"Top {top_n} labels by accuracy: {top_labels}")

# Compute probabilities for ROC curve plotting
# MultiOutputClassifier returns a list of arrays - we need to stack them
y_prob = np.array([pred[:, 1] for pred in clf.predict_proba(X_test)]).T


# Settings
background = shap.utils.sample(X_train, 5)  # Background samples
test_samples = X_test[:5]  # Test samples to explain
max_display = 20  # Max features to show
sample_to_explain = 0  # Index for waterfall plots

# Loop through labels
for label_name in top_labels:
    label_idx = y.columns.get_loc(label_name)
    print(f"\nComputing SHAP for: {label_name}")

    predict_fn = lambda X, idx=label_idx: clf.predict_proba(X)[idx][:, 1]
    explainer = shap.KernelExplainer(predict_fn, background)

    shap_values = np.array(explainer.shap_values(test_samples))

    if len(shap_values.shape) == 1:
        shap_values = shap_values.reshape(1, -1)

    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    sorted_idx = np.argsort(-mean_abs_shap)[:max_display]
    sorted_features = X.columns[sorted_idx]
    sorted_values = mean_abs_shap[sorted_idx]

    plt.figure(figsize=(12, 6))
    plt.barh(range(len(sorted_idx)), sorted_values[::-1],
             color='skyblue', edgecolor='black')
    plt.yticks(range(len(sorted_idx)), sorted_features[::-1])
    plt.xlabel("Mean |SHAP Value| (Average Impact on Model Output)")
    plt.title(f"Global Feature Importance for '{label_name}'", pad=20)
    plt.tight_layout()
    plt.savefig(f'shap_{label_name}_top.png')
    plt.show()


