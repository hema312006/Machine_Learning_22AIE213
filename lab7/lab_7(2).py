import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# classifiers
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

# metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ---------------------------------
# Load dataset
# ---------------------------------
df = pd.read_excel("Conf_Text_Labels.xlsx")

df = df.dropna(subset=['Text', 'Conf Label'])
df['Text'] = df['Text'].astype(str)
df['Conf Label'] = df['Conf Label'].astype(int)

X = df['Text']
y = df['Conf Label']


# ---------------------------------
# Train Test Split
# ---------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ---------------------------------
# Classifiers
# ---------------------------------
models = {
    "SVM": LinearSVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Naive Bayes": MultinomialNB(),
    "AdaBoost": AdaBoostClassifier(),
    "MLP": MLPClassifier(max_iter=300)
}


# ---------------------------------
# Results storage
# ---------------------------------
results = []


# ---------------------------------
# Train & Evaluate each model
# ---------------------------------
for name, model in models.items():

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', model)
    ])

    # train
    pipeline.fit(X_train, y_train)

    # predictions
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    # metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    precision = precision_score(y_test, y_test_pred, average='weighted')
    recall = recall_score(y_test, y_test_pred, average='weighted')
    f1 = f1_score(y_test, y_test_pred, average='weighted')

    results.append([name, train_acc, test_acc, precision, recall, f1])


# ---------------------------------
# Create Results Table
# ---------------------------------
results_df = pd.DataFrame(results, columns=[
    "Model",
    "Train Accuracy",
    "Test Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
])

print(results_df)
