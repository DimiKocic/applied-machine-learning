from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import shap
from functools import partial
import arff
from sklearn.decomposition import PCA
import joblib

# Commented for performance issues
# # Read ARFF file
# with open("IMDB-F.arff", 'r') as f:
#     dataset = arff.load(f)
#
# Convert to pandas DataFrame
# df = pd.DataFrame(dataset['data'], columns=[attr[0] for attr in dataset['attributes']])

df = pd.read_csv(r'.\NeuralNetwork\IMDB-F_CSV.csv', header=0, sep=',')

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

# Create and train the neural network classifier
mlp = MLPClassifier(hidden_layer_sizes=(256,128), max_iter=1000,verbose=True, random_state=42, activation='relu',
                         solver='adam',
                         learning_rate_init=0.01,
                         alpha=0.01,
                         batch_size=64)

# Wrap it in a MultiOutputClassifier for multi-label classification
# Commented for performance issues
# multi_output_mlp = MultiOutputClassifier(mlp)
# Train the model with PCA-transformed features
# multi_output_mlp.fit(X_train_pca, y_train)
# joblib.dump(multi_output_mlp, 'multi_output_mlp_model.pkl')

# Load a saved file
multi_output_mlp = joblib.load(r'.\NeuralNetwork\multi_output_mlp_model.pkl')
y_pred = multi_output_mlp.predict(X_test_pca)
# Evaluate the performance
print("Classification Report for Each Label:")
print(classification_report(y_test, y_pred, target_names=y.columns,zero_division=1))

# Confusion Matrix
cm = confusion_matrix(y_test.values.argmax(axis=1), y_pred.argmax(axis=1))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=y.columns)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# Get predicted probabilities
y_score = multi_output_mlp.predict_proba(X_test_pca)

# Binarize the test output (if it's not already binary)
y_test_bin = y_test.values

# Plot ROC curve for each label
fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = y_test_bin.shape[1]
plt.figure(figsize=(10, 8))

for i in range(n_classes):
    # Extract the predicted probabilities for class i
    y_prob = y_score[i][:, 1]
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob)
    roc_auc[i] = auc(fpr[i], tpr[i])

    plt.plot(fpr[i], tpr[i], lw=2,
             label=f'{y.columns[i]} (AUC = {roc_auc[i]:.2f})')

# Plot random line
plt.plot([0, 1], [0, 1], 'k--', lw=2)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Multi-label Classification')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=300)  # You can change the filename or dpi as needed
plt.show()


mlp_for_shap = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=100, random_state=42,
                             activation='relu', solver='adam', alpha=0.01,
                             learning_rate_init=0.01, batch_size=64, verbose=True)


# Commented for performance issues
# multi_output_mlp_for_shap = MultiOutputClassifier(mlp_for_shap)
# multi_output_mlp_for_shap.fit(X_train, y_train)
# joblib.dump(multi_output_mlp_for_shap, "multi_output_mlp_for_shap2.pkl")

# Load a saved File
multi_output_mlp_for_shap = joblib.load(r'.\NeuralNetwork\multi_output_mlp_for_shap2.pkl')

# Convert to DataFrames to keep feature names
X_train_df = pd.DataFrame(X_train, columns=X.columns)
X_test_df = pd.DataFrame(X_test, columns=X.columns)

# Number of top features to show
TOP_N_FEATURES = 10  # Adjust this number as needed


# Function to predict probabilities for one class
def predict_fn_for_class(class_idx, X):
    return multi_output_mlp_for_shap.predict_proba(X)[class_idx][:, 1]


# Create SHAP explainers for each label
for i, label in enumerate(y.columns):
    print(f"Calculating SHAP values for {label}")

    # Create explainer for this label
    explainer = shap.KernelExplainer(
        partial(predict_fn_for_class, i),
        shap.sample(X_train_df, 20)  # Background data
    )

    # Calculate SHAP values for a sample of test data
    test_sample = X_test_df.iloc[:20]  # Using more samples for better stability
    shap_values = explainer.shap_values(test_sample)

    # Get mean absolute SHAP values for each feature
    mean_shap_values = np.abs(shap_values).mean(axis=0)

    # Create a DataFrame for visualization
    shap_df = pd.DataFrame({
        'Feature': X_test_df.columns,
        'SHAP_Value': mean_shap_values
    })

    # Sort and get top features
    top_shap_df = shap_df.sort_values('SHAP_Value', ascending=False).head(TOP_N_FEATURES)

    # Plot horizontal bar chart
    plt.figure(figsize=(10, 6))
    plt.barh(top_shap_df['Feature'], top_shap_df['SHAP_Value'], color='#1f77b4')
    plt.gca().invert_yaxis()  # Highest value at top
    plt.title(f'Top {TOP_N_FEATURES} Features by SHAP Value - {label}')
    plt.xlabel('Mean |SHAP Value| (impact on model output)')
    plt.tight_layout()
    plt.savefig(f'Top {TOP_N_FEATURES} Features by SHAP Value - {label}.png')
