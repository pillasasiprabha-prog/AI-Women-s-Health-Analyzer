import streamlit as st
import sqlite3
import pandas as pd
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# --- PAGE CONFIG ---
st.set_page_config(page_title="Women's Health AI", page_icon="🌸", layout="wide")

# --- DATABASE INITIALIZATION ---
def init_db():
    conn = sqlite3.connect("womens_health_v2.db", check_same_thread=False)
    cursor = conn.cursor()
    cursor.executescript("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE, name TEXT, created_at TEXT
    );
    CREATE TABLE IF NOT EXISTS health_records(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_email TEXT, problem TEXT, solution TEXT, recorded_at TEXT
    );
    """)
    conn.commit()
    return conn

conn = init_db()

# --- ML MODEL TRAINING (Using your health_data) ---
# Data from your source code
health_data = {
    "problem": [
        "stress anxiety mental health", "weight gain obesity overweight",
        "weight loss underweight thin", "pcod pcos polycystic ovary",
        "irregular periods menstruation", "pregnancy prenatal maternal",
        "hair loss thinning hairfall", "skin acne pimple dark spots",
        "body pain joint muscle ache", "thyroid hypothyroid",
        "diabetes blood sugar", "anaemia iron deficiency"
    ],
    "solution": [
        "Meditation, Yoga, Therapy, Journaling",
        "Low-calorie diet, 45min cardio daily",
        "High-protein diet, strength training",
        "Hormone therapy, low-GI diet, exercise",
        "Gynecologist checkup, iron-rich diet",
        "Prenatal vitamins, regular OB-GYN visits",
        "Biotin, protein-rich diet, gentle care",
        "Hydration, SPF, Vitamin C, gentle cleanser",
        "Physiotherapy, anti-inflammatory diet",
        "Thyroid medication, iodine-rich diet",
        "Low-GI diet, regular monitoring, exercise",
        "Iron & B12 supplements, spinach, legumes"
    ]
}

@st.cache_resource
def train_model():
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("clf", MultinomialNB(alpha=0.5))
    ])
    pipeline.fit(health_data["problem"], health_data["solution"])
    return pipeline

model = train_model()

# --- APP UI ---
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Symptom Checker", "Health Records", "About ML Model"])

    if page == "Home":
        st.title("🌸 Women's Health & Wellness AI")
        st.markdown("""
        Welcome to your personal health assistant. This application uses **Natural Language Processing** to analyze your symptoms and provide preliminary wellness advice.
        """)
        st.info("💡 Note: This tool is for educational purposes and is not a substitute for professional medical advice.")
        

    elif page == "Symptom Checker":
        st.header("Analyze Your Symptoms")
        user_input = st.text_area("Describe how you are feeling (e.g., 'I am having irregular periods and feeling stressed'):")
        
        if st.button("Generate Health Report"):
            if user_input:
                # ML Prediction
                prediction = model.predict([user_input.lower()])[0]
                
                # Display Results
                st.subheader("Results")
                st.success(f"**Potential Path:** {prediction}")
                
                # Save to DB
                cursor = conn.cursor()
                cursor.execute("INSERT INTO health_records (problem, solution, recorded_at) VALUES (?, ?, ?)", 
                               (user_input, prediction, datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
                conn.commit()
                
                st.info("Record saved to history.")
            else:
                st.warning("Please enter your symptoms first.")

    elif page == "Health Records":
        st.header("Consultation History")
        df = pd.read_sql_query("SELECT recorded_at as Date, problem as 'Symptoms', solution as 'Advice' FROM health_records", conn)
        if not df.empty:
            st.dataframe(df, use_container_width=True)
        else:
            st.write("No records found.")

    elif page == "About ML Model":
        st.header("Technical Specifications")
        st.write("**Algorithm:** Multinomial Naive Bayes")
        st.write("**Vectorization:** TF-IDF with Bigrams")
        st.write("**Accuracy:** High (Specific to trained dataset)")
        

#Image of Naive Bayes classifier algorithm diagram


if __name__ == "__main__":
    main()
