from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import joblib

# ... (prepare the dataset with features and target variable as described above)
# Load data from the Excel file
df = pd.read_excel("output_dataframe.xlsx")

df = df[~((df['Title'] == '') | (df['Title'] == 'Newsbeast'))]
print(df)

# Separate features (X) and target variable (y)
X = df[['Title length', 'Title Sentiment', 'Article_Text_Length', 'Article Text Sentiment', 
'Title-Text Sentiment Similarity', 'Title-Text Similarity', 'Curiosity Gap', 'Numbered List']]

y = df['Clickbait Score']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42)

# Train the model
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report_str)

# Get feature importances
feature_importances = rf_classifier.feature_importances_

# Create a DataFrame to store the feature importances along with their names
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})

# Sort the DataFrame by importance in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Print the feature importances
print(feature_importance_df)

# You can also visualize the feature importances using a bar plot
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

# Assuming feature_importance_df has columns 'Feature' and 'Importance'

plt.figure(figsize=(10, 6))

# Bar plot
bars = plt.bar(feature_importance_df['Feature'], feature_importance_df['Importance'], color='black', alpha=0.7)

# Add labels and title
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importances')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Add value annotations on top of each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 4), ha='center', va='bottom')

# Display the plot
plt.tight_layout()
plt.show()

# Save the trained classifier to a file
model_filename = 'trained_model.pkl'
joblib.dump(rf_classifier, model_filename)