from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def train_model():
    X = ["stress anxiety", "weight gain"]
    y = ["meditation", "diet"]

    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", MultinomialNB())
    ])

    model.fit(X, y)
    return model

model = train_model()

def predict(problem):
    return model.predict([problem])[0]
