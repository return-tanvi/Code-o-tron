import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Assume you have a much larger dataset loaded into a pandas DataFrame
# For demonstration, let's create a larger synthetic dataset
data = {'notes': [], 'label': []}
for i in range(1000):
    if i % 5 == 0:
        data['notes'].append(f'Dangerous situation leading to a potential {["trip and fall", "collision", "burn"][i % 3]}')
        data['label'].append('unsafe')
    else:
        data['notes'].append(f'Just some {["funny", "cute", "interesting"][i % 3]} content with {["upbeat music", "nice visuals", "a simple tutorial"][i % 3]}')
        data['label'].append('safe')

df = pd.DataFrame(data)

# Convert labels to numerical values
df['label_numeric'] = df['label'].apply(lambda x: 1 if x == 'unsafe' else 0)

# Split data into training and testing sets (e.g., 80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(df['notes'], df['label_numeric'], test_size=0.2, random_state=42)

# Feature extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train a Logistic Regression model
model = LogisticRegression(random_state=42)
model.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on the larger test set: {accuracy:.2f}")

print("\nClassification Report on the larger test set:")
print(classification_report(y_test, y_pred))

# Example of predicting on new data
new_notes = pd.Series(['Someone slipped on a wet floor', 'A dog happily chasing its tail'])
new_notes_tfidf = tfidf_vectorizer.transform(new_notes)
new_predictions = model.predict(new_notes_tfidf)

print("\nPredictions on new data:")
for note, prediction in zip(new_notes, new_predictions):
    predicted_label = 'unsafe' if prediction == 1 else 'safe'
    print(f"Note: '{note}' - Predicted Label: {predicted_label}")