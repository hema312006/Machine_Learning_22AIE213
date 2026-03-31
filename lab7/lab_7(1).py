import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

df = pd.read_excel("Conf_Text_Labels.xlsx")

# removing missing
df = df.dropna(subset=['Text', 'Conf Label'])

# converting text to string (IMPORTANT FIX)
df['Text'] = df['Text'].astype(str)

# removing blank text
df = df[df['Text'].str.strip() != ""]

# converting label to int
df['Conf Label'] = df['Conf Label'].astype(int)

X = df['Text']
y = df['Conf Label']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

def tune_random_forest(X_train, y_train):

    # Pipeline: TF-IDF + RandomForest
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('rf', RandomForestClassifier(random_state=42))
    ])

    # Hyperparameter search space
    param_dist = {
        "tfidf__max_features": [1000, 3000],
        "tfidf__ngram_range": [(1,1), (1,2)],
        "rf__n_estimators": [50, 100, 150],
        "rf__max_depth": [None, 10, 20],
        "rf__min_samples_split": [2, 5, 10]
    }

    # RandomizedSearchCV
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=5,
        cv=3,
        random_state=42,
        n_jobs=1
    )

    random_search.fit(X_train, y_train)

    return random_search.best_estimator_, random_search.best_params_

best_model, best_params = tune_random_forest(X_train, y_train)

print("Best Parameters:", best_params)
