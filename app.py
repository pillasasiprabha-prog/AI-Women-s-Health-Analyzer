import tkinter as tk
import sqlite3
import hashlib
import datetime
import random
import sys

# ── optional heavy deps (graceful fallback) ──
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        confusion_matrix, classification_report
    )
    import numpy as np
    HAS_SK = True
except ImportError:
    HAS_SK = False

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    HAS_RL = True
except ImportError:
    HAS_RL = False

# ══════════════════════════════════════════════
#                    THEME
# ══════════════════════════════════════════════
BG        = "#0f0a1e"
BG2       = "#160d28"
CARD      = "#1e1235"
CARD2     = "#261848"
ACCENT    = "#c084fc"
ACCENT2   = "#f0abfc"
ACCENT3   = "#818cf8"
TEXT      = "#f3e8ff"
MUTED     = "#9d7ec9"
SUCCESS   = "#4ade80"
WARNING   = "#facc15"
DANGER    = "#f87171"
INFO      = "#60a5fa"
ENTRY_BG  = "#2a1650"
BTN_BG    = "#7c3aed"
BTN_HOV   = "#9333ea"
BTN_OK    = "#059669"
BTN_OK_H  = "#10b981"
BORDER    = "#3b2060"

FONT      = ("Segoe UI", 10)
FONT_B    = ("Segoe UI", 10, "bold")
FONT_H    = ("Segoe UI", 14, "bold")
FONT_T    = ("Segoe UI", 20, "bold")
FONT_LOGO = ("Segoe UI", 28, "bold")
FONT_SM   = ("Segoe UI", 9)
FONT_MONO = ("Consolas", 9)

# ══════════════════════════════════════════════
#                   DATABASE
# ══════════════════════════════════════════════
conn   = sqlite3.connect("womens_health_v2.db")
cursor = conn.cursor()
cursor.executescript("""
CREATE TABLE IF NOT EXISTS users(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE, password TEXT,
    name TEXT, age TEXT, address TEXT, blood_group TEXT,
    height REAL, weight REAL, created_at TEXT
);
CREATE TABLE IF NOT EXISTS health_records(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_email TEXT, problem TEXT, solution TEXT,
    bmi REAL, bmi_category TEXT, notes TEXT, recorded_at TEXT
);
CREATE TABLE IF NOT EXISTS appointments(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_email TEXT, doctor_name TEXT, specialty TEXT,
    date TEXT, time TEXT, notes TEXT, status TEXT
);
CREATE TABLE IF NOT EXISTS cycle_tracker(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_email TEXT, period_start TEXT,
    period_end TEXT, cycle_length INTEGER, notes TEXT
);
CREATE TABLE IF NOT EXISTS feedback(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_email TEXT, text TEXT, rating INTEGER, created_at TEXT
);
""")
conn.commit()

# ══════════════════════════════════════════════
#                   ML MODEL
# ══════════════════════════════════════════════
health_data = {
    "problem": [
        "stress anxiety mental health", "weight gain obesity overweight",
        "weight loss underweight thin", "pcod pcos polycystic ovary",
        "irregular periods menstruation", "pregnancy prenatal maternal",
        "hair loss thinning hairfall", "skin acne pimple dark spots",
        "body pain joint muscle ache", "thyroid hypothyroid",
        "diabetes blood sugar", "anaemia iron deficiency",
        "migraine headache", "insomnia sleep disorder",
        "back pain spine posture", "depression mood low",
        "high blood pressure hypertension", "digestive bloating constipation",
        "breast health mammogram", "bone density osteoporosis",
        "menopause hot flashes", "urinary infection uti",
        "endometriosis pelvic pain", "postpartum depression new mother",
        "childhood nutrition growing girl", "teen puberty adolescent",
        # Augmented samples for better ML training (duplicates with variation)
        "stress work anxiety panic attacks", "obesity bmi high sedentary",
        "thin weak low weight malnourished", "pcos hormones irregular cycle",
        "period delay missed menstrual cycle", "prenatal baby maternity care",
        "hair thinning scalp bald patches", "acne breakout oily skin face",
        "pain joints muscles arthritis", "thyroid gland metabolism slow",
        "sugar high insulin resistance glucose", "iron low haemoglobin pale",
        "headache migraine aura nausea", "poor sleep insomnia restless",
        "spine lower back pain sciatica", "sad mood depression low energy",
        "blood pressure high stroke risk", "gut bloating ibs constipation",
        "breast lump cancer screening", "weak bones fracture osteo",
        "menopause perimenopause hormones", "uti bladder frequent urination",
        "endometriosis cramps pelvic", "postpartum baby blues mood",
        "child growth nutrition vitamins", "puberty teen girl adolescence"
    ],
    "solution": [
        "Meditation, Yoga, Therapy, Journaling, Breathing exercises",
        "Low-calorie diet, 45min cardio daily, Reduce sugar/processed food",
        "High-protein diet, strength training, calorie surplus, frequent meals",
        "Hormone therapy, low-GI diet, regular exercise, stress reduction",
        "Gynecologist checkup, iron-rich diet, track cycle, avoid stress",
        "Prenatal vitamins, regular OB-GYN visits, balanced nutrition, rest",
        "Biotin supplements, scalp massage, protein-rich diet, gentle hair care",
        "Hydration, SPF, Vitamin C serum, gentle cleanser, balanced diet",
        "Physiotherapy, hot/cold therapy, anti-inflammatory diet, rest",
        "Thyroid medication, iodine-rich diet, regular TSH tests",
        "Low-GI diet, regular monitoring, exercise, medication if needed",
        "Iron & B12 supplements, spinach, legumes, red meat (if non-veg)",
        "Dark room rest, hydration, magnesium supplement, avoid triggers",
        "Sleep hygiene, no screens before bed, chamomile tea, melatonin",
        "Core strengthening, ergonomic posture, physio, hot compress",
        "Therapy/counselling, social support, exercise, medication if needed",
        "Low-sodium diet, regular BP monitoring, aerobic exercise, medication",
        "Probiotics, fibre-rich diet, hydration, reduce dairy/gluten",
        "Monthly self-exam, annual mammogram, healthy weight, no smoking",
        "Calcium + Vitamin D, weight-bearing exercise, bone density scan",
        "HRT if needed, cooling techniques, soy isoflavones, yoga",
        "Hydration, D-Mannose, cranberry extract, antibiotics if severe",
        "Pain management, hormonal therapy, laparoscopy, pelvic physio",
        "Therapy, partner support, medication, rest, gentle walks",
        "Calcium-rich foods, balanced macros, limit junk, regular activity",
        "Iron-rich foods, calcium, hygiene education, body positivity",
        # Augmented labels (same solution classes)
        "Meditation, Yoga, Therapy, Journaling, Breathing exercises",
        "Low-calorie diet, 45min cardio daily, Reduce sugar/processed food",
        "High-protein diet, strength training, calorie surplus, frequent meals",
        "Hormone therapy, low-GI diet, regular exercise, stress reduction",
        "Gynecologist checkup, iron-rich diet, track cycle, avoid stress",
        "Prenatal vitamins, regular OB-GYN visits, balanced nutrition, rest",
        "Biotin supplements, scalp massage, protein-rich diet, gentle hair care",
        "Hydration, SPF, Vitamin C serum, gentle cleanser, balanced diet",
        "Physiotherapy, hot/cold therapy, anti-inflammatory diet, rest",
        "Thyroid medication, iodine-rich diet, regular TSH tests",
        "Low-GI diet, regular monitoring, exercise, medication if needed",
        "Iron & B12 supplements, spinach, legumes, red meat (if non-veg)",
        "Dark room rest, hydration, magnesium supplement, avoid triggers",
        "Sleep hygiene, no screens before bed, chamomile tea, melatonin",
        "Core strengthening, ergonomic posture, physio, hot compress",
        "Therapy/counselling, social support, exercise, medication if needed",
        "Low-sodium diet, regular BP monitoring, aerobic exercise, medication",
        "Probiotics, fibre-rich diet, hydration, reduce dairy/gluten",
        "Monthly self-exam, annual mammogram, healthy weight, no smoking",
        "Calcium + Vitamin D, weight-bearing exercise, bone density scan",
        "HRT if needed, cooling techniques, soy isoflavones, yoga",
        "Hydration, D-Mannose, cranberry extract, antibiotics if severe",
        "Pain management, hormonal therapy, laparoscopy, pelvic physio",
        "Therapy, partner support, medication, rest, gentle walks",
        "Calcium-rich foods, balanced macros, limit junk, regular activity",
        "Iron-rich foods, calcium, hygiene education, body positivity"
    ]
}

# ── Short label map for confusion matrix (to keep it readable) ──
SHORT_LABEL_MAP = {
    "Meditation, Yoga, Therapy, Journaling, Breathing exercises":          "Stress/Anxiety",
    "Low-calorie diet, 45min cardio daily, Reduce sugar/processed food":   "Wt Gain",
    "High-protein diet, strength training, calorie surplus, frequent meals":"Wt Loss",
    "Hormone therapy, low-GI diet, regular exercise, stress reduction":     "PCOS",
    "Gynecologist checkup, iron-rich diet, track cycle, avoid stress":      "Irreg Period",
    "Prenatal vitamins, regular OB-GYN visits, balanced nutrition, rest":   "Pregnancy",
    "Biotin supplements, scalp massage, protein-rich diet, gentle hair care":"Hair",
    "Hydration, SPF, Vitamin C serum, gentle cleanser, balanced diet":      "Skin/Acne",
    "Physiotherapy, hot/cold therapy, anti-inflammatory diet, rest":        "Body Pain",
    "Thyroid medication, iodine-rich diet, regular TSH tests":              "Thyroid",
    "Low-GI diet, regular monitoring, exercise, medication if needed":      "Diabetes",
    "Iron & B12 supplements, spinach, legumes, red meat (if non-veg)":     "Anaemia",
    "Dark room rest, hydration, magnesium supplement, avoid triggers":      "Migraine",
    "Sleep hygiene, no screens before bed, chamomile tea, melatonin":       "Insomnia",
    "Core strengthening, ergonomic posture, physio, hot compress":          "Back Pain",
    "Therapy/counselling, social support, exercise, medication if needed":  "Depression",
    "Low-sodium diet, regular BP monitoring, aerobic exercise, medication": "Hypert.",
    "Probiotics, fibre-rich diet, hydration, reduce dairy/gluten":          "Digestive",
    "Monthly self-exam, annual mammogram, healthy weight, no smoking":      "Breast",
    "Calcium + Vitamin D, weight-bearing exercise, bone density scan":      "Osteo",
    "HRT if needed, cooling techniques, soy isoflavones, yoga":             "Menopause",
    "Hydration, D-Mannose, cranberry extract, antibiotics if severe":       "UTI",
    "Pain management, hormonal therapy, laparoscopy, pelvic physio":        "Endometrio.",
    "Therapy, partner support, medication, rest, gentle walks":             "Postpartum",
    "Calcium-rich foods, balanced macros, limit junk, regular activity":    "Child Nutr.",
    "Iron-rich foods, calcium, hygiene education, body positivity":         "Teen Health",
}

# Global metrics store (computed once)
_ML_METRICS = {}

if HAS_SK:
    import pandas as _pd
    df_health = _pd.DataFrame(health_data)
    X = df_health["problem"].tolist()
    y = df_health["solution"].tolist()

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("clf",   MultinomialNB(alpha=0.5))
    ])
    pipeline.fit(X, y)

    # ── Compute ML metrics via cross-validation ──
    try:
        unique_classes = list(dict.fromkeys(y))  # preserve order, deduplicate
        n_classes = len(unique_classes)
        # Use 5-fold CV; if too few samples fall back to leave-one-out style
        n_splits = min(5, len(X) // n_classes) if len(X) // n_classes >= 2 else 2
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        y_pred_cv = cross_val_predict(pipeline, X, y, cv=cv)

        _ML_METRICS["accuracy"]    = round(accuracy_score(y, y_pred_cv) * 100, 2)
        _ML_METRICS["precision"]   = round(precision_score(y, y_pred_cv, average="weighted", zero_division=0) * 100, 2)
        _ML_METRICS["recall"]      = round(recall_score(y, y_pred_cv, average="weighted", zero_division=0) * 100, 2)
        _ML_METRICS["f1"]          = round(f1_score(y, y_pred_cv, average="weighted", zero_division=0) * 100, 2)
        _ML_METRICS["cm"]          = confusion_matrix(y, y_pred_cv, labels=unique_classes)
        _ML_METRICS["labels"]      = [SHORT_LABEL_MAP.get(c, c[:12]) for c in unique_classes]
        _ML_METRICS["full_labels"] = unique_classes
        _ML_METRICS["report"]      = classification_report(y, y_pred_cv, target_names=_ML_METRICS["labels"], zero_division=0)
        _ML_METRICS["n_samples"]   = len(X)
        _ML_METRICS["n_classes"]   = n_classes
        _ML_METRICS["algo"]        = "Multinomial Naive Bayes + TF-IDF (1-2 grams)"
        _ML_METRICS["cv_folds"]    = n_splits
    except Exception as _me:
        _ML_METRICS["error"] = str(_me)

    def predict_solution(problem):
        return pipeline.predict([problem.lower()])[0]
else:
    _sol_map = dict(zip(health_data["problem"], health_data["solution"]))
    def predict_solution(problem):
        p = problem.lower()
        for k, v in _sol_map.items():
            if any(w in p for w in k.split()):
                return v
        return "Consult a doctor for personalised advice."

# ══════════════════════════════════════════════
#           COMPREHENSIVE HEALTH DATA
# ══════════════════════════════════════════════
HEALTH_DETAILS = {
    "stress / anxiety": {
        "age_groups": "All ages (especially teens, working women, new mothers)",
        "causes": [
            "Work/academic pressure", "Relationship conflicts",
            "Financial concerns", "Hormonal imbalances (PMS, menopause)",
            "Trauma or past abuse", "Social media overexposure"
        ],
        "symptoms": [
            "Constant worry or overthinking", "Rapid heartbeat, sweating",
            "Headaches and muscle tension", "Fatigue and poor concentration",
            "Irritability, mood swings", "Sleep disturbances"
        ],
        "precautions": [
            "Limit news/social media to 30 min/day",
            "Practice box breathing (4-4-4-4 counts)",
            "Set healthy boundaries at work and home",
            "Keep a gratitude journal nightly",
            "Talk to a trusted friend or therapist"
        ],
        "diet_plan": [
            "🍌 Bananas & dark chocolate – mood boosters (serotonin)",
            "🫐 Blueberries – antioxidants fight stress hormones",
            "🥑 Avocado – B vitamins for nervous system",
            "🌿 Green tea – L-theanine for calm focus",
            "🐟 Fatty fish (salmon) – omega-3 reduces cortisol",
            "🥜 Almonds & walnuts – magnesium & healthy fats",
            "❌ Avoid caffeine, alcohol, excessive sugar"
        ],
        "yoga_poses": [
            "🧘 Child's Pose (Balasana) – 5 min",
            "🧘 Legs-Up-The-Wall (Viparita Karani) – 10 min",
            "🧘 Cat-Cow stretch – 3 min",
            "🧘 Corpse Pose (Savasana) – 10 min",
            "🧘 Alternate Nostril Breathing (Nadi Shodhana)"
        ],
        "medicines": [
            "Ashwagandha (adaptogen supplement)",
            "Magnesium Glycinate 300mg",
            "B-Complex vitamins",
            "⚕ Prescribed: SSRIs or anxiolytics (consult psychiatrist)"
        ],
        "doctors": ["Psychiatrist / Psychologist", "Gynaecologist (if hormonal)", "General Physician"]
    },
    "weight gain / obesity": {
        "age_groups": "All ages; post-pregnancy, menopause, PCOD-related",
        "causes": [
            "Sedentary lifestyle", "Hormonal imbalances (thyroid, PCOD)",
            "Emotional eating", "Sleep deprivation",
            "Medications (steroids, antidepressants)", "Insulin resistance"
        ],
        "symptoms": [
            "BMI above 25 (overweight) or 30 (obese)",
            "Fatigue with minimal activity", "Joint pain (knees, hips)",
            "Snoring / sleep apnoea", "Irregular periods", "Low self-esteem"
        ],
        "precautions": [
            "Track calories with a food diary (target 500 kcal deficit)",
            "Never skip breakfast – fuels metabolism",
            "Drink 500 ml water before each meal",
            "Eat slowly; chew 20–30 times per bite",
            "Sleep 7–8 hrs; sleep deprivation increases ghrelin"
        ],
        "diet_plan": [
            "🥗 Plate rule: ½ veggies, ¼ protein, ¼ complex carbs",
            "🍗 Lean proteins: chicken, tofu, lentils, eggs",
            "🥦 Non-starchy veggies: broccoli, spinach, zucchini",
            "🍓 Low-sugar fruits: berries, apple, pear",
            "🌾 Complex carbs: oats, quinoa, brown rice",
            "❌ Cut: sugary drinks, fried snacks, white bread, alcohol"
        ],
        "yoga_poses": [
            "🔥 Sun Salutation (Surya Namaskar) – 12 rounds/day",
            "🔥 Warrior I & II – tones legs & core",
            "🔥 Boat Pose (Navasana) – core strength",
            "🔥 Bridge Pose – activates glutes & metabolism",
            "🔥 Twisted Chair Pose – detox & toning"
        ],
        "medicines": [
            "Vitamin D3 + K2 (often deficient in obesity)",
            "Omega-3 fish oil 1000mg",
            "Probiotics for gut health",
            "⚕ Prescribed: Orlistat / Metformin (if insulin resistant)"
        ],
        "doctors": ["Nutritionist / Dietitian", "Endocrinologist", "Bariatric Specialist (if BMI > 35)"]
    },
    "weight loss / underweight": {
        "age_groups": "Teens, young adults, post-illness recovery",
        "causes": [
            "Inadequate caloric intake", "Hyperthyroidism",
            "Eating disorders (anorexia, bulimia)",
            "Malabsorption / IBS / Crohn's disease",
            "Depression or chronic stress", "Parasitic infections"
        ],
        "symptoms": [
            "BMI below 18.5", "Fatigue and weakness",
            "Brittle nails and hair loss", "Frequent illness (low immunity)",
            "Irregular or absent periods", "Poor wound healing"
        ],
        "precautions": [
            "Eat every 3–4 hours; never skip meals",
            "Add healthy calorie-dense snacks between meals",
            "Avoid excessive cardio – focus on strength training",
            "Rule out thyroid / digestive disorders with blood tests",
            "Address emotional causes with a therapist if needed"
        ],
        "diet_plan": [
            "🥛 Full-fat dairy: milk, Greek yoghurt, paneer",
            "🥜 Nuts & nut butters: almonds, peanut butter, cashews",
            "🍌 High-calorie fruits: banana, mango, avocado, dates",
            "🍚 Complex carbs: rice, whole wheat roti, sweet potato",
            "🥚 Eggs + legumes – protein for muscle gain",
            "🫒 Healthy oils: olive oil, ghee in cooking",
            "✅ Aim: 300–500 kcal surplus above daily needs"
        ],
        "yoga_poses": [
            "💪 Warrior III – builds muscle & balance",
            "💪 Chair Pose (Utkatasana) – lower body strength",
            "💪 Cobra Pose – spinal strength, digestion",
            "💪 Shoulder Stand (Sarvangasana) – thyroid stimulation",
            "💪 Downward Dog – full-body toning"
        ],
        "medicines": [
            "Protein supplements (whey / plant-based) post-workout",
            "Multivitamin with zinc & iron",
            "Vitamin B12 (especially if vegetarian)",
            "⚕ If thyroid: consult endocrinologist for medication"
        ],
        "doctors": ["Nutritionist / Dietitian", "Endocrinologist", "Gastroenterologist", "Psychiatrist (if eating disorder)"]
    },
    "pcod / pcos": {
        "age_groups": "Teens to women aged 12–45",
        "causes": [
            "Insulin resistance (most common cause)",
            "Elevated androgens (male hormones)",
            "Genetic predisposition", "Chronic inflammation",
            "Sedentary lifestyle + poor diet", "Stress and sleep deprivation"
        ],
        "symptoms": [
            "Irregular or absent periods", "Excess facial / body hair (hirsutism)",
            "Acne and oily skin", "Weight gain (especially belly fat)",
            "Hair thinning on scalp", "Difficulty conceiving",
            "Dark skin patches (acanthosis nigricans)", "Mood swings"
        ],
        "precautions": [
            "Lose even 5–10% body weight – restores periods in many women",
            "Avoid refined sugar and white carbohydrates completely",
            "Practise seed cycling (flax+pumpkin days 1–14; sesame+sunflower days 15–28)",
            "Check fasting insulin, testosterone, AMH every 6 months",
            "Use non-comedogenic skincare products"
        ],
        "diet_plan": [
            "🌾 Low-GI foods: oats, quinoa, millet, barley",
            "🥦 Anti-inflammatory veggies: broccoli, kale, spinach",
            "🍒 Berries + cherries – lower insulin spikes",
            "🐟 Fatty fish 3x/week – reduce inflammation",
            "🫘 Lentils, chickpeas – plant protein + fibre",
            "🌰 Spearmint tea – reduces androgen levels naturally",
            "❌ Avoid: dairy excess, white sugar, processed carbs, soy"
        ],
        "yoga_poses": [
            "🌸 Butterfly Pose (Baddha Konasana) – opens hips",
            "🌸 Reclined Butterfly – relaxes pelvic organs",
            "🌸 Garland Pose (Malasana) – improves circulation",
            "🌸 Supported Bridge – balances hormones",
            "🌸 Supta Matsyendrasana – detoxes abdominal organs"
        ],
        "medicines": [
            "Myo-Inositol 2000mg + D-Chiro-Inositol 50mg (evidence-based)",
            "Vitamin D3 2000–4000 IU (most PCOS women are deficient)",
            "Omega-3 fatty acids 1–2g/day",
            "Spearmint capsules / tea (anti-androgenic)",
            "⚕ Prescribed: Metformin, OCP, Clomiphene (for fertility)"
        ],
        "doctors": ["Gynaecologist / Reproductive Endocrinologist", "Endocrinologist", "Nutritionist", "Dermatologist (for skin/hair)"]
    },
    "pregnancy / prenatal care": {
        "age_groups": "Women aged 18–45 (pregnancy and postpartum)",
        "causes": ["Normal physiological process; complications from nutrition, stress, infections"],
        "symptoms": [
            "1st Trimester: nausea, fatigue, breast tenderness, frequent urination",
            "2nd Trimester: visible bump, fetal movements, back pain",
            "3rd Trimester: heartburn, breathlessness, swollen feet, Braxton Hicks"
        ],
        "precautions": [
            "Start folic acid 400mcg at least 1 month before conception",
            "Avoid alcohol, smoking, raw fish, unpasteurised dairy",
            "Attend all scheduled antenatal checkups",
            "Sleep on left side – improves fetal blood flow",
            "Avoid heavy lifting after 20 weeks",
            "Monitor fetal movements daily from 28 weeks"
        ],
        "diet_plan": [
            "🥛 Dairy: 3 servings/day for calcium (baby's bones)",
            "🥦 Dark leafy greens: folate, iron, calcium",
            "🍊 Citrus fruits: Vitamin C enhances iron absorption",
            "🥚 Eggs: choline for fetal brain development",
            "🐟 Safe fish: salmon, sardines (DHA for brain)",
            "💧 Water: at least 2.5 litres/day",
            "❌ Avoid: raw eggs, liver (excess Vit A), papaya, pineapple"
        ],
        "yoga_poses": [
            "🤰 Cat-Cow (gentle, all trimesters)",
            "🤰 Warrior II (1st & 2nd trimester only)",
            "🤰 Prenatal Child's Pose (modified)",
            "🤰 Seated Forward Bend (gentle)",
            "🤰 Kegel exercises (pelvic floor strength)",
            "⚠ Always practise with certified prenatal yoga instructor"
        ],
        "medicines": [
            "Folic Acid 400–800 mcg (prevent neural tube defects)",
            "Iron 27mg + Vitamin C (prevent anaemia)",
            "Calcium 1000mg + Vitamin D3",
            "DHA / Omega-3 (fetal brain development)",
            "⚕ Prescribed: Iron infusion, anti-emetics if needed"
        ],
        "doctors": ["Obstetrician / Gynaecologist", "Nutritionist", "Physiotherapist (pelvic floor)", "Paediatrician (newborn care)"]
    },
    "irregular periods": {
        "age_groups": "Teens, reproductive age women, perimenopause",
        "causes": [
            "PCOS / PCOD", "Thyroid disorders (hypo or hyper)",
            "Stress and anxiety", "Sudden weight gain or loss",
            "Over-exercise", "Perimenopause", "Uterine fibroids/polyps"
        ],
        "symptoms": [
            "Cycles shorter than 21 days or longer than 35 days",
            "Missed periods (oligomenorrhoea or amenorrhoea)",
            "Very heavy or very light flow", "Severe cramping",
            "Spotting between periods", "PMS symptoms intensified"
        ],
        "precautions": [
            "Track period with an app (days, flow, symptoms)",
            "Manage stress – cortisol directly suppresses reproductive hormones",
            "Maintain healthy BMI (extremes both disrupt periods)",
            "Get tested: thyroid, prolactin, testosterone, FSH, LH",
            "Avoid extreme diets or over-exercising"
        ],
        "diet_plan": [
            "🌿 Chasteberry (Vitex) tea – regulates LH naturally",
            "🥬 Iron-rich foods during heavy flow: spinach, beetroot",
            "🌾 Seed cycling protocol throughout cycle",
            "🍫 Dark chocolate 70%+ – magnesium reduces cramps",
            "🌿 Ginger tea – reduces prostaglandins (cramping)",
            "❌ Avoid: excess soy, processed foods, caffeine"
        ],
        "yoga_poses": [
            "🌸 Reclining Bound Angle Pose",
            "🌸 Seated Forward Fold – stimulates ovaries",
            "🌸 Cobra Pose – hormonal regulation",
            "🌸 Head-to-Knee Pose (Janu Sirsasana)",
            "🌸 Shoulder Stand (stimulates thyroid & ovaries)"
        ],
        "medicines": [
            "Vitamin D3 + K2", "Magnesium 300–400mg",
            "Chasteberry / Vitex extract",
            "Omega-3 fatty acids",
            "⚕ Prescribed: Hormonal therapy, OCP, Progesterone"
        ],
        "doctors": ["Gynaecologist", "Endocrinologist", "Nutritionist"]
    },
    "hair problems": {
        "age_groups": "Teens to post-menopausal women",
        "causes": [
            "Nutritional deficiencies (iron, biotin, zinc, protein)",
            "PCOS / hormonal imbalance", "Thyroid disorders",
            "Post-pregnancy hormonal drop (telogen effluvium)",
            "Chemical treatments / heat styling", "Alopecia areata (autoimmune)"
        ],
        "symptoms": [
            "Excessive hair on pillow / in shower drain",
            "Thinning at temples or crown", "Receding hairline",
            "Brittle, dry, or frizzy hair", "Scalp itching or dandruff",
            "Loss of more than 100 hairs/day"
        ],
        "precautions": [
            "Blood test: iron, ferritin, B12, D3, thyroid, testosterone",
            "Use wide-tooth comb on wet hair; never brush wet",
            "Avoid tight hairstyles (ponytails, braids) daily",
            "Oil scalp 2x/week (coconut, castor, onion oil)",
            "Switch to sulphate-free, gentle shampoo"
        ],
        "diet_plan": [
            "🥚 Eggs: biotin + protein – #1 hair food",
            "🌿 Spinach: iron, folate, Vitamins A & C",
            "🥜 Almonds & walnuts: biotin, Vitamin E, zinc",
            "🐟 Fatty fish: omega-3 for scalp health",
            "🫘 Lentils: protein + biotin + iron combo",
            "🌸 Amla (Indian gooseberry): Vitamin C richest food",
            "❌ Avoid: crash diets, excess Vitamin A supplements"
        ],
        "yoga_poses": [
            "💆 Downward Dog – increases scalp blood flow",
            "💆 Headstand (Sirsasana) – if experienced practitioner",
            "💆 Rabbit Pose (Sasangasana) – stimulates scalp",
            "💆 Camel Pose – balances thyroid",
            "💆 Kapalbhati pranayama 10 min daily"
        ],
        "medicines": [
            "Biotin 5000–10000 mcg/day",
            "Iron (Ferrous Gluconate) if ferritin < 70",
            "Vitamin D3 2000 IU", "Zinc 25–50mg",
            "Saw Palmetto (if DHT-related loss)",
            "⚕ Topical: Minoxidil 2–5% (prescribed)"
        ],
        "doctors": ["Dermatologist / Trichologist", "Endocrinologist", "Gynaecologist (if hormonal)"]
    },
    "skin problems / acne": {
        "age_groups": "Teens, young adults, hormonal acne in 20s–30s",
        "causes": [
            "Excess sebum production", "Bacterial overgrowth (C. acnes)",
            "Hormonal changes (PCOS, puberty, menstrual cycle)",
            "Dairy and high-GI foods", "Stress (cortisol spikes)",
            "Wrong skincare products", "Pollution / sun damage"
        ],
        "symptoms": [
            "Blackheads, whiteheads, cysts", "Red inflamed pimples",
            "Post-acne dark spots (PIH)", "Oily T-zone",
            "Dry patches + breakouts (combination skin)",
            "Skin texture and enlarged pores"
        ],
        "precautions": [
            "Never pick or squeeze pimples – worsens scarring",
            "Change pillowcase every 3–4 days",
            "Apply SPF 30+ every morning (non-comedogenic)",
            "Remove makeup fully before sleeping",
            "Keep hair products away from skin"
        ],
        "diet_plan": [
            "💧 Water 3L/day – flushes toxins",
            "🥦 Zinc-rich foods: pumpkin seeds, chickpeas",
            "🫐 Antioxidant berries – fight inflammation",
            "🌿 Green tea (drink + apply as toner)",
            "🐟 Omega-3: salmon, flaxseeds – reduce inflammation",
            "❌ Avoid: dairy, white sugar, whey protein, fried food",
            "❌ Limit: high-GI foods (white rice, bread, sweets)"
        ],
        "yoga_poses": [
            "✨ Pranayama (breathing) – oxygenates blood",
            "✨ Forward bends – improve circulation to face",
            "✨ Fish Pose – stimulates thyroid, improves skin",
            "✨ Twisting poses – liver detox",
            "✨ Shoulder Stand – hormonal balance"
        ],
        "medicines": [
            "Niacinamide 4–10% serum (topical)",
            "Salicylic Acid 2% cleanser",
            "Zinc supplement 30–50mg",
            "Vitamin C serum + SPF",
            "⚕ Prescribed: Clindamycin, Retinoids, Isotretinoin, OCP"
        ],
        "doctors": ["Dermatologist", "Gynaecologist (if hormonal acne)", "Nutritionist"]
    },
    "thyroid problems": {
        "age_groups": "Women aged 20–60 (5–8x more common in women than men)",
        "causes": [
            "Autoimmune (Hashimoto's – hypothyroid; Graves' – hyperthyroid)",
            "Iodine deficiency or excess", "Radiation therapy history",
            "Genetic predisposition", "Pregnancy (postpartum thyroiditis)",
            "Certain medications (lithium, amiodarone)"
        ],
        "symptoms": [
            "Hypothyroid: fatigue, weight gain, cold intolerance, constipation, dry skin",
            "Hyperthyroid: weight loss, heat intolerance, rapid heartbeat, anxiety",
            "Both: hair loss, irregular periods, mood changes",
            "Goitre (enlarged thyroid gland)"
        ],
        "precautions": [
            "Test TSH, T3, T4, TPO antibodies annually",
            "Take thyroid medication on empty stomach, 30–60 min before food",
            "Don't take with calcium, iron or antacids – blocks absorption",
            "Manage stress – cortisol suppresses thyroid function",
            "Avoid excessive raw goitrogenic foods (broccoli, cabbage)"
        ],
        "diet_plan": [
            "🐟 Seaweed / seafood: iodine for hypothyroid",
            "🥩 Lean meats + eggs: selenium (Brazil nuts too)",
            "🌾 Gluten-free if Hashimoto's confirmed",
            "🫘 Avoid unfermented soy – blocks iodine uptake",
            "🌿 Ashwagandha: adaptogen that supports thyroid",
            "❌ Raw goitrogens in excess: cabbage, cauliflower (cook them)"
        ],
        "yoga_poses": [
            "🦋 Shoulder Stand (Sarvangasana) – directly stimulates thyroid",
            "🦋 Fish Pose (Matsyasana) – stretches thyroid area",
            "🦋 Plough Pose (Halasana)",
            "🦋 Camel Pose – opens throat chakra",
            "🦋 Lion's Breath – stimulates thyroid gland"
        ],
        "medicines": [
            "Levothyroxine (T4 replacement) – hypothyroid",
            "Selenium 200mcg (supports conversion T4→T3)",
            "Vitamin D3 + B12 (commonly deficient)",
            "Zinc 25mg",
            "⚕ Hyperthyroid: Methimazole, Radioiodine, Surgery"
        ],
        "doctors": ["Endocrinologist", "Gynaecologist (if fertility affected)", "Nutritionist"]
    },
    "diabetes / blood sugar": {
        "age_groups": "Gestational: pregnant women; Type 2: 30+; Type 1: any age",
        "causes": [
            "Insulin resistance (Type 2)", "Autoimmune destruction of beta cells (Type 1)",
            "Pregnancy hormones (gestational)", "PCOS-related insulin resistance",
            "Obesity, sedentary lifestyle", "Genetic predisposition"
        ],
        "symptoms": [
            "Frequent thirst and urination", "Unexplained fatigue",
            "Blurry vision", "Slow-healing wounds",
            "Tingling in feet/hands", "Frequent infections (UTI, yeast)"
        ],
        "precautions": [
            "Monitor fasting blood sugar and HbA1c every 3 months",
            "Never skip meals – eat at regular intervals",
            "Check feet daily for cuts, sores, or numbness",
            "Exercise for 30 min daily – lowers blood sugar naturally",
            "Test before and after meals to understand food responses"
        ],
        "diet_plan": [
            "🌾 Low-GI grains: oats, barley, quinoa, millets",
            "🥦 Non-starchy vegetables at every meal",
            "🫘 Legumes: chickpeas, lentils, rajma",
            "🍎 Low-GI fruits: berries, guava, apple, pear",
            "🥩 Protein with every meal: slows glucose absorption",
            "🌿 Cinnamon 1 tsp/day: lowers blood sugar naturally",
            "❌ Avoid: white rice, sugary drinks, sweets, potatoes, fruit juice"
        ],
        "yoga_poses": [
            "🔄 Bow Pose (Dhanurasana) – stimulates pancreas",
            "🔄 Seated Forward Fold – calms nervous system",
            "🔄 Supine Spinal Twist – massages abdominal organs",
            "🔄 Legs Up the Wall – improves circulation",
            "🔄 Kapalbhati + Anulom Vilom pranayama"
        ],
        "medicines": [
            "Chromium Picolinate 400mcg (improves insulin sensitivity)",
            "Berberine 500mg (comparable to Metformin in studies)",
            "Magnesium Glycinate 300mg", "Alpha-Lipoic Acid 600mg",
            "⚕ Prescribed: Metformin, Insulin, GLP-1 agonists"
        ],
        "doctors": ["Endocrinologist / Diabetologist", "Nutritionist", "Cardiologist", "Ophthalmologist"]
    },
    "anaemia / iron deficiency": {
        "age_groups": "Most common in women of reproductive age, teen girls, pregnant women",
        "causes": [
            "Heavy menstrual bleeding", "Poor dietary iron intake",
            "Pregnancy (increased iron demand)", "Malabsorption (celiac, IBS)",
            "Vegetarian/vegan diet without planning", "Chronic inflammation"
        ],
        "symptoms": [
            "Extreme fatigue and weakness", "Pale skin and conjunctiva",
            "Shortness of breath on exertion", "Cold hands and feet",
            "Brittle nails / koilonychia (spoon nails)", "Restless legs at night",
            "Difficulty concentrating (brain fog)"
        ],
        "precautions": [
            "Test: CBC, Serum Ferritin, Serum Iron, TIBC",
            "Take iron supplement on empty stomach with Vitamin C",
            "Never take iron with calcium, tea, or coffee",
            "Cook in cast iron pan – transfers dietary iron to food",
            "Address root cause (heavy periods → see gynaecologist)"
        ],
        "diet_plan": [
            "🥬 Spinach + lemon juice – iron + absorption enhancer",
            "🫘 Rajma, masoor dal, chickpeas – plant iron sources",
            "🥩 Red meat 2–3x/week (if non-vegetarian) – heme iron",
            "🌿 Beetroot + carrot juice daily",
            "🍊 Vitamin C with every iron-rich meal",
            "🌰 Pumpkin seeds, sesame seeds – iron-rich snacks",
            "❌ Avoid: tea/coffee with meals (tannins block iron)"
        ],
        "yoga_poses": [
            "❤️ Legs Up the Wall – improves blood circulation",
            "❤️ Supported Bridge Pose – energising",
            "❤️ Seated Forward Fold – calming, restorative",
            "❤️ Gentle Cobra – stimulates digestive organs",
            "❤️ Deep belly breathing – oxygenation"
        ],
        "medicines": [
            "Ferrous Gluconate or Ferrous Bisglycinate (gentler on stomach)",
            "Vitamin C 500mg with each iron dose",
            "B12 (methylcobalamin 1000mcg) if also B12 deficient",
            "Folate / Folic Acid 400–800mcg",
            "⚕ Severe: IV Iron infusion or Blood transfusion"
        ],
        "doctors": ["Gynaecologist", "Haematologist", "Gastroenterologist (if malabsorption)", "Nutritionist"]
    },
    "depression / mood disorders": {
        "age_groups": "Any age; peaks in postpartum, perimenopause, adolescence",
        "causes": [
            "Serotonin, dopamine, norepinephrine imbalance",
            "Hormonal fluctuations (postpartum, PMDD, menopause)",
            "Thyroid dysfunction", "Chronic stress or trauma",
            "Vitamin D, B12, omega-3 deficiency", "Social isolation"
        ],
        "symptoms": [
            "Persistent sadness or emptiness", "Loss of interest in activities",
            "Fatigue and sleep disturbances (too much or too little)",
            "Appetite changes – weight gain or loss",
            "Difficulty concentrating", "Feelings of worthlessness",
            "Thoughts of self-harm (seek emergency help immediately)"
        ],
        "precautions": [
            "Seek professional help – depression is a medical condition",
            "Don't isolate – reach out to 1 trusted person daily",
            "Establish routine: wake, eat, sleep at consistent times",
            "Exercise is as effective as antidepressants for mild depression",
            "Limit alcohol – it worsens depressive symptoms"
        ],
        "diet_plan": [
            "🐟 Fatty fish 3x/week: omega-3 (EPA/DHA) – brain health",
            "🍫 Dark chocolate: phenylethylamine boosts mood",
            "🫐 Berries: polyphenols protect brain cells",
            "🌿 Turmeric: curcumin as effective as antidepressants in studies",
            "🥬 Leafy greens: folate for serotonin synthesis",
            "🍌 Banana + yoghurt: tryptophan → serotonin",
            "❌ Avoid: alcohol, ultra-processed foods, excess sugar"
        ],
        "yoga_poses": [
            "🌞 Sun Salutation – energises, boosts serotonin",
            "🌞 Backbends (Camel, Fish, Cobra) – heart openers",
            "🌞 Inversions: Legs Up the Wall – shifts perspective",
            "🌞 Yoga Nidra (body scan meditation) – 20 min",
            "🌞 Dance / movement therapy (Nritya Yoga)"
        ],
        "medicines": [
            "Omega-3 (EPA 1000mg+) – reduces depression symptoms",
            "Vitamin D3 2000–4000 IU (test levels first)",
            "St. John's Wort (mild–moderate; drug interactions – consult doctor)",
            "Saffron extract (clinical studies support mood benefits)",
            "⚕ Prescribed: SSRIs, SNRIs, Therapy (CBT/DBT)"
        ],
        "doctors": ["Psychiatrist", "Psychologist / Therapist", "Endocrinologist (if hormonal)", "Neurologist"]
    },
    "menopause": {
        "age_groups": "Women aged 45–55 (perimenopause may start at 40)",
        "causes": ["Natural decline in oestrogen and progesterone production", "Surgical menopause (after oophorectomy)"],
        "symptoms": [
            "Hot flashes and night sweats", "Irregular then absent periods",
            "Vaginal dryness", "Mood swings, anxiety, depression",
            "Insomnia", "Weight gain (especially belly)",
            "Reduced libido", "Brain fog", "Joint pain"
        ],
        "precautions": [
            "Keep bedroom cool (18–19°C) for night sweats",
            "Wear breathable natural fabrics",
            "Avoid triggers: spicy food, caffeine, alcohol",
            "Do weight-bearing exercise to protect bone density",
            "Maintain vaginal health with moisturisers and lubricants"
        ],
        "diet_plan": [
            "🫘 Soy isoflavones: tofu, edamame, miso (phytoestrogens)",
            "🥛 Calcium: dairy, fortified plant milk, almonds",
            "🐟 Fatty fish: omega-3 for heart & mood",
            "🌾 Flaxseeds: lignans balance hormones",
            "🍎 Fibre-rich foods prevent weight gain",
            "❌ Limit: alcohol, caffeine, sugary foods (worsen hot flashes)"
        ],
        "yoga_poses": [
            "🌙 Yin Yoga: long-held, cooling poses",
            "🌙 Moon Salutation (Chandra Namaskar)",
            "🌙 Seated Wide-Angle Forward Fold",
            "🌙 Restorative Yoga: bolster-supported poses",
            "🌙 Shavasana with guided relaxation"
        ],
        "medicines": [
            "Soy isoflavones 40–80mg (non-hormonal option)",
            "Black Cohosh extract",
            "Vitamin D3 2000 IU + Calcium 1200mg",
            "Magnesium Glycinate (sleep + mood)",
            "⚕ HRT: Oestrogen/Progesterone therapy (discuss risks with doctor)"
        ],
        "doctors": ["Gynaecologist / Menopause Specialist", "Endocrinologist", "Cardiologist", "Bone Density Specialist"]
    },
    "migraine / headache": {
        "age_groups": "Women 3x more likely than men; peaks ages 20–45",
        "causes": [
            "Hormonal changes (oestrogen fluctuations around periods)",
            "Stress and sleep disruption",
            "Dietary triggers: tyramine, MSG, alcohol, caffeine withdrawal",
            "Sensory triggers: bright light, strong smells",
            "Dehydration", "Weather/pressure changes"
        ],
        "symptoms": [
            "Throbbing pain (usually one side)", "Nausea and vomiting",
            "Light and sound sensitivity", "Aura: visual disturbances, tingling",
            "Neck stiffness before attack", "Cognitive fog (migraine hangover)"
        ],
        "precautions": [
            "Keep a migraine diary (identify patterns and triggers)",
            "Sleep consistent hours – even weekends",
            "Stay hydrated: 2–3 litres water daily",
            "Avoid skipping meals – hypoglycaemia triggers migraine",
            "Wear sunglasses in bright light / use blue-light filters"
        ],
        "diet_plan": [
            "💧 Water first on waking – dehydration is #1 trigger",
            "🥜 Magnesium-rich: almonds, pumpkin seeds, dark leafy greens",
            "🐟 Omega-3: reduce frequency of migraines",
            "🍒 Cherries / tart cherry juice – anti-inflammatory",
            "☕ Small amount of caffeine can help during attack",
            "❌ Avoid: red wine, aged cheese, processed meats, MSG, aspartame"
        ],
        "yoga_poses": [
            "🧊 Child's Pose with forehead on block",
            "🧊 Legs Up the Wall (during attack – dark room)",
            "🧊 Supported Reclining Bound Angle",
            "🧊 Neck and shoulder stretches (prevent tension migraines)",
            "🧊 Alternate Nostril Breathing"
        ],
        "medicines": [
            "Magnesium Glycinate 400mg daily (preventive)",
            "Riboflavin (B2) 400mg daily (preventive)",
            "CoQ10 300mg daily (preventive)",
            "Melatonin 3mg at bedtime",
            "⚕ Prescribed: Triptans (acute), Topiramate/Propranolol (preventive)"
        ],
        "doctors": ["Neurologist", "Gynaecologist (if menstrual migraine)", "Ophthalmologist", "Pain Management Specialist"]
    },
    "back pain / posture": {
        "age_groups": "All ages; most common in working women, pregnant women, post-delivery",
        "causes": [
            "Prolonged sitting with poor posture", "Weak core muscles",
            "Heavy lifting (including newborn care)", "Pregnancy-related postural shift",
            "Disc herniation or sciatica", "Osteoporosis in older women"
        ],
        "symptoms": [
            "Dull aching lower or upper back pain", "Pain after sitting long hours",
            "Radiating pain to legs (sciatica)", "Morning stiffness",
            "Pain worsening with bending or lifting", "Poor posture (rounded shoulders)"
        ],
        "precautions": [
            "Set screen at eye level; chair should support lumbar curve",
            "Stand and stretch for 2 min every 30 min",
            "Lift with bent knees, not bent back",
            "Sleep on firm mattress; side-lie with pillow between knees",
            "Strengthen core before starting exercise programs"
        ],
        "diet_plan": [
            "🐟 Omega-3: salmon, walnuts – reduce spinal inflammation",
            "🥛 Calcium: dairy, broccoli, almonds – bone strength",
            "🌿 Turmeric + ginger: natural anti-inflammatories",
            "🍒 Cherries: collagen and anti-inflammatory",
            "💧 Adequate hydration – maintains disc fluid balance",
            "❌ Avoid: processed foods, excess sugar (pro-inflammatory)"
        ],
        "yoga_poses": [
            "🌿 Cat-Cow Stretch – spinal mobility",
            "🌿 Child's Pose – lower back release",
            "🌿 Supine Twist – relieves sciatica",
            "🌿 Pigeon Pose – hip flexor stretch",
            "🌿 Bridge Pose – strengthens lower back & glutes",
            "🌿 Downward Dog – decompresses spine"
        ],
        "medicines": [
            "Magnesium Glycinate (muscle relaxant)",
            "Turmeric Curcumin 500–1000mg",
            "Vitamin D3 + K2 (bone health)",
            "Topical: Diclofenac gel",
            "⚕ Prescribed: Muscle relaxants, NSAIDs, Physiotherapy, Injections"
        ],
        "doctors": ["Physiotherapist", "Orthopaedic Surgeon", "Spine Specialist", "Chiropractor", "Neurologist (if sciatica)"]
    },
    "childhood / teen health (girls)": {
        "age_groups": "Girls aged 8–18 (puberty to late teens)",
        "causes": ["Nutritional deficiencies during growth spurts", "Body image issues", "Academic stress", "Social pressure"],
        "symptoms": [
            "Irregular periods in first 2 years (normal)", "Acne during puberty",
            "Growth spurts and bone pain", "Mood swings (hormonal)",
            "Low energy / fatigue (often iron deficiency)", "Body image concerns"
        ],
        "precautions": [
            "Ensure calcium + Vitamin D for bone building (peak bone mass by 25)",
            "Educate on menstrual hygiene and cycle tracking",
            "Screen time limits – affects sleep and mental health",
            "Encourage sport and physical activity",
            "Open conversations about body changes and mental wellbeing"
        ],
        "diet_plan": [
            "🥛 Dairy or fortified plant milk: calcium for growing bones",
            "🥚 Eggs + dal: protein for growth and hormones",
            "🥬 Iron-rich foods: spinach, rajma (onset of menstruation increases needs)",
            "🍊 Citrus daily: Vitamin C + immunity",
            "🌰 Healthy snacks: nuts, seeds, fruits instead of chips",
            "❌ Limit: sugary drinks, energy drinks, junk food"
        ],
        "yoga_poses": [
            "🌈 Tree Pose (balance & focus)",
            "🌈 Warrior poses (confidence & strength)",
            "🌈 Forward folds (calm exam anxiety)",
            "🌈 Happy Baby Pose (relaxation)",
            "🌈 Dance yoga / fun movement classes"
        ],
        "medicines": [
            "Multivitamin with iron for teen girls",
            "Calcium 1300mg + Vitamin D3 600 IU",
            "Omega-3 (DHA/EPA) for brain development",
            "B-Complex for stress and energy",
            "⚕ Consult paediatrician for personalised guidance"
        ],
        "doctors": ["Paediatrician / Adolescent Medicine", "Gynaecologist (first visit by 13–15)", "Nutritionist", "School Counsellor"]
    },
}

PROBLEM_MAP = {
    "Stress / Anxiety":                "stress / anxiety",
    "Weight Gain / Obesity":           "weight gain / obesity",
    "Weight Loss / Underweight":       "weight loss / underweight",
    "PCOD / PCOS":                     "pcod / pcos",
    "Pregnancy / Prenatal Care":       "pregnancy / prenatal care",
    "Irregular Periods":               "irregular periods",
    "Hair Problems":                   "hair problems",
    "Skin Problems / Acne":            "skin problems / acne",
    "Thyroid Problems":                "thyroid problems",
    "Diabetes / Blood Sugar":          "diabetes / blood sugar",
    "Anaemia / Iron Deficiency":       "anaemia / iron deficiency",
    "Depression / Mood Disorders":     "depression / mood disorders",
    "Menopause":                       "menopause",
    "Migraine / Headache":             "migraine / headache",
    "Back Pain / Posture":             "back pain / posture",
    "Childhood / Teen Health (Girls)": "childhood / teen health (girls)",
}

# ══════════════════════════════════════════════
#                  HELPERS
# ══════════════════════════════════════════════
def hash_pw(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def bmi_calc(weight, height_cm):
    if height_cm <= 0: return 0, "N/A"
    h   = height_cm / 100
    bmi = round(weight / (h * h), 1)
    if bmi < 18.5:  cat = "Underweight"
    elif bmi < 25:  cat = "Normal"
    elif bmi < 30:  cat = "Overweight"
    else:           cat = "Obese"
    return bmi, cat

def styled_btn(parent, text, cmd, bg=BTN_BG, fg=TEXT, width=18, pad=8):
    hov = BTN_HOV if bg == BTN_BG else bg
    b = tk.Button(parent, text=text, command=cmd,
                  bg=bg, fg=fg, font=FONT_B, relief="flat",
                  bd=0, cursor="hand2", width=width,
                  padx=pad, pady=7,
                  activebackground=hov, activeforeground=TEXT)
    b.bind("<Enter>", lambda e: b.config(bg=hov))
    b.bind("<Leave>", lambda e: b.config(bg=bg))
    return b

def entry(parent, show=None, width=32):
    return tk.Entry(parent, bg=ENTRY_BG, fg=TEXT, font=FONT,
                    insertbackground=ACCENT, relief="flat",
                    width=width, show=show)

def card(parent, **kw):
    return tk.Frame(parent, bg=CARD, bd=0, relief="flat", **kw)

def label(parent, text, font=FONT, fg=TEXT, bg=None, **kw):
    return tk.Label(parent, text=text, font=font,
                    fg=fg, bg=bg or parent.cget("bg"), **kw)

def section_title(parent, text):
    f = tk.Frame(parent, bg=CARD)
    f.pack(fill="x", padx=18, pady=(14, 4))
    tk.Label(f, text=text, font=FONT_B, fg=ACCENT2, bg=CARD).pack(side="left")
    tk.Frame(f, bg=BORDER, height=1).pack(side="left", fill="x", expand=True, padx=(8, 0), pady=6)

def bullet_list(parent, items, icon="•", fg=TEXT):
    for item in items:
        row = tk.Frame(parent, bg=CARD)
        row.pack(fill="x", padx=28, pady=1)
        tk.Label(row, text=icon, fg=ACCENT, bg=CARD, font=FONT, width=2).pack(side="left")
        tk.Label(row, text=item, fg=fg, bg=CARD, font=FONT,
                 anchor="w", wraplength=520, justify="left").pack(side="left", fill="x", expand=True)

def make_scrollable(parent, bg=BG):
    wrap = tk.Frame(parent, bg=bg)
    wrap.pack(fill="both", expand=True)
    canvas = tk.Canvas(wrap, bg=bg, highlightthickness=0)
    sb     = ttk.Scrollbar(wrap, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=sb.set)
    sb.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
    inner  = tk.Frame(canvas, bg=bg)
    wid    = canvas.create_window((0, 0), window=inner, anchor="nw")
    inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.bind("<Configure>", lambda e: canvas.itemconfig(wid, width=e.width))
    canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
    return inner

# ══════════════════════════════════════════════
#                  NAVIGATION
# ══════════════════════════════════════════════
class NavBar(tk.Frame):
    def __init__(self, master, app):
        super().__init__(master, bg=CARD2, height=40)
        self.pack(fill="x", side="top")
        for w in [tk.Frame(self, bg=ACCENT, width=4, height=40)]:
            w.pack(side="left", fill="y")

        brand = tk.Label(self, text="✦ WellnessAI", font=("Segoe UI", 11, "bold"),
                         fg=ACCENT, bg=CARD2)
        brand.pack(side="left", padx=10)

        nav_items = [
            ("🏠", DashboardPage), ("📋", DetailsPage), ("🔬", AnalysisPage),
            ("⚖", BMIPage), ("🌸", CycleTrackerPage),
            ("📅", AppointmentPage), ("📂", HistoryPage),
            ("📊", MLMetricsPage),   # ← NEW
        ]
        for icon, page in nav_items:
            b = tk.Button(self, text=icon, bg=CARD2, fg=MUTED,
                          font=("Segoe UI", 13), relief="flat", cursor="hand2",
                          padx=6, pady=5, activebackground=BG, activeforeground=ACCENT,
                          command=lambda p=page: app.show_frame(p))
            b.pack(side="left")

        tk.Button(self, text="🚪 Logout", bg=CARD2, fg=DANGER,
                  font=FONT_SM, relief="flat", cursor="hand2",
                  padx=8, pady=5,
                  command=lambda: app.show_frame(LoginPage)).pack(side="right", padx=4)

# ══════════════════════════════════════════════
#                   MAIN APP
# ══════════════════════════════════════════════
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("✦ AI Women's Health Analyser")
        self.geometry("960x720")
        self.minsize(820, 640)
        self.configure(bg=BG)
        self.resizable(True, True)
        self.current_user = {}

        self.frames = {}
        pages = [LoginPage, RegisterPage, DashboardPage, DetailsPage,
                 AnalysisPage, BMIPage, CycleTrackerPage,
                 AppointmentPage, HistoryPage, FeedbackPage, MLMetricsPage]
        for F in pages:
            f = F(self)
            self.frames[F] = f
            f.place(relwidth=1, relheight=1)

        self.show_frame(LoginPage)

    def show_frame(self, page):
        self.frames[page].tkraise()
        if hasattr(self.frames[page], "on_show"):
            self.frames[page].on_show()

# ══════════════════════════════════════════════
#            PAGE 1 – LOGIN
# ══════════════════════════════════════════════
class LoginPage(tk.Frame):
    def __init__(self, master):
        super().__init__(master, bg=BG)
        self._build()

    def _build(self):
        tk.Frame(self, bg=ACCENT, height=4).pack(fill="x")
        center = tk.Frame(self, bg=BG)
        center.pack(expand=True)

        tk.Label(center, text="✦", font=("Segoe UI", 44), fg=ACCENT, bg=BG).pack(pady=(40, 0))
        tk.Label(center, text="AI Women's Health", font=FONT_LOGO, fg=TEXT, bg=BG).pack()
        tk.Label(center, text="Your personalised wellness companion",
                 font=FONT, fg=MUTED, bg=BG).pack(pady=(2, 24))

        c = card(center)
        c.pack(ipadx=20, ipady=10, padx=60, fill="x")

        tk.Label(c, text="Email Address", font=FONT_B, fg=ACCENT2, bg=CARD).pack(pady=(20, 4))
        self.email = entry(c)
        self.email.pack()

        tk.Label(c, text="Password", font=FONT_B, fg=ACCENT2, bg=CARD).pack(pady=(12, 4))
        self.pw = entry(c, show="●")
        self.pw.pack()

        styled_btn(c, "Login  →", self.login, width=22).pack(pady=18)

        sep = tk.Frame(c, bg=BORDER, height=1)
        sep.pack(fill="x", padx=20)

        tk.Label(c, text="New here?", font=FONT_SM, fg=MUTED, bg=CARD).pack(pady=(12, 4))
        styled_btn(c, "Create Account", lambda: self.master.show_frame(RegisterPage),
                   bg="#1e1040", width=22).pack(pady=(0, 20))

        quotes = [
            "\"Your health is your wealth.\"",
            "\"Self-care is not selfish.\"",
            "\"Strong women lift each other up.\"",
        ]
        tk.Label(self, text=random.choice(quotes),
                 font=("Segoe UI", 9, "italic"), fg=MUTED, bg=BG).pack(pady=10)

    def login(self):
        email = self.email.get().strip()
        pw    = self.pw.get()
        if not email or not pw:
            messagebox.showwarning("Missing Fields", "Enter email and password.")
            return
        cursor.execute("SELECT * FROM users WHERE email=? AND password=?",
                       (email, hash_pw(pw)))
        row = cursor.fetchone()
        if row:
            self.master.current_user = {
                "id": row[0], "email": row[1], "name": row[3],
                "age": row[4], "address": row[5], "blood_group": row[6],
                "height": row[7], "weight": row[8]
            }
            self.master.show_frame(DashboardPage)
        else:
            messagebox.showerror("Login Failed", "Invalid email or password.")

# ══════════════════════════════════════════════
#            PAGE 2 – REGISTER
# ══════════════════════════════════════════════
class RegisterPage(tk.Frame):
    def __init__(self, master):
        super().__init__(master, bg=BG)
        self._master = master
        self._build()

    def _build(self):
        tk.Frame(self, bg=ACCENT, height=4).pack(fill="x")
        tk.Label(self, text="Create Your Account", font=FONT_T, fg=TEXT, bg=BG).pack(pady=(16, 2))
        tk.Label(self, text="Fill all details to register", font=FONT_SM, fg=MUTED, bg=BG).pack()

        inner = make_scrollable(self)
        c = card(inner)
        c.pack(padx=80, pady=10, fill="x")

        fields = [
            ("📧 Email",         "email",       False),
            ("🔒 Password",      "password",    True),
            ("👤 Full Name",     "name",        False),
            ("🎂 Age",           "age",         False),
            ("🏠 Address",       "address",     False),
            ("🩸 Blood Group",   "blood_group", False),
            ("📏 Height (cm)",   "height",      False),
            ("⚖  Weight (kg)",   "weight",      False),
        ]
        self.entries = {}
        for lbl, key, secret in fields:
            tk.Label(c, text=lbl, font=FONT_B, fg=ACCENT, bg=CARD).pack(pady=(12, 2))
            e = entry(c, show="●" if secret else None)
            e.pack()
            self.entries[key] = e

        btn_f = tk.Frame(c, bg=CARD)
        btn_f.pack(pady=20)
        styled_btn(btn_f, "✅  Register", self._register, width=20).pack(pady=6)
        styled_btn(btn_f, "← Back to Login", self._back, bg="#1e1040", width=20).pack()

    def _back(self):
        self._master.show_frame(LoginPage)

    def _register(self):
        vals = {k: v.get().strip() for k, v in self.entries.items()}
        empty = [k for k, v in vals.items() if not v]
        if empty:
            messagebox.showwarning("Missing", f"Fill: {', '.join(empty)}")
            return
        if len(vals["password"]) < 6:
            messagebox.showwarning("Weak Password", "Minimum 6 characters.")
            return
        try:
            h, w = float(vals["height"]), float(vals["weight"])
        except ValueError:
            messagebox.showerror("Invalid", "Height and Weight must be numbers.")
            return
        try:
            cursor.execute(
                "INSERT INTO users(email,password,name,age,address,blood_group,height,weight,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                (vals["email"], hash_pw(vals["password"]), vals["name"],
                 vals["age"], vals["address"], vals["blood_group"], h, w,
                 datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
            conn.commit()
            messagebox.showinfo("Success ✅", f"Welcome, {vals['name']}!\nPlease login.")
            for e in self.entries.values(): e.delete(0, tk.END)
            self._master.show_frame(LoginPage)
        except sqlite3.IntegrityError:
            messagebox.showerror("Already Registered", "Email already in use.")
        except Exception as ex:
            messagebox.showerror("Error", str(ex))

# ══════════════════════════════════════════════
#            DASHBOARD
# ══════════════════════════════════════════════
class DashboardPage(tk.Frame):
    def __init__(self, master):
        super().__init__(master, bg=BG)

    def on_show(self):
        for w in self.winfo_children(): w.destroy()
        NavBar(self, self.master)

        user  = self.master.current_user
        email = user.get("email", "")
        name  = user.get("name", "User")

        hdr = tk.Frame(self, bg=CARD2)
        hdr.pack(fill="x", padx=0, pady=0)
        tk.Label(hdr, text=f"Welcome back, {name}  💜",
                 font=FONT_T, fg=ACCENT, bg=CARD2).pack(pady=(12, 2))
        tk.Label(hdr, text=datetime.datetime.now().strftime("%A, %d %B %Y"),
                 font=FONT_SM, fg=MUTED, bg=CARD2).pack(pady=(0, 12))

        cursor.execute("SELECT COUNT(*) FROM health_records WHERE user_email=?", (email,))
        rec = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM appointments WHERE user_email=?", (email,))
        appt = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM cycle_tracker WHERE user_email=?", (email,))
        cyc = cursor.fetchone()[0]
        h = float(user.get("height") or 0)
        w = float(user.get("weight") or 0)
        bmi, cat = bmi_calc(w, h) if h and w else (0, "N/A")

        stats = [("📋 Records", str(rec), ACCENT), ("📅 Appts", str(appt), ACCENT2),
                 ("🌸 Cycles", str(cyc), SUCCESS), ("⚖ BMI", f"{bmi}", WARNING)]
        sf = tk.Frame(self, bg=BG)
        sf.pack(fill="x", padx=20, pady=12)
        for t, v, col in stats:
            b = card(sf)
            b.pack(side="left", expand=True, fill="both", padx=6)
            tk.Label(b, text=t,  font=FONT_SM, fg=MUTED,   bg=CARD).pack(pady=(10, 0))
            tk.Label(b, text=v,  font=FONT_H,  fg=col,     bg=CARD).pack()
            tk.Label(b, text=cat if t == "⚖ BMI" else "",
                     font=FONT_SM, fg=MUTED, bg=CARD).pack(pady=(0, 10))

        banner = card(self)
        banner.pack(fill="x", padx=20, pady=4)
        tk.Label(banner, text="✦  5-Step Health Journey",
                 font=FONT_H, fg=ACCENT2, bg=CARD).pack(pady=(12, 4))

        steps = [
            ("1", "Create Account", LoginPage),
            ("2", "Your Details",   DetailsPage),
            ("3", "AI Analysis",    AnalysisPage),
            ("4", "Solutions",      AnalysisPage),
            ("5", "Feedback",       FeedbackPage),
        ]
        sf2 = tk.Frame(banner, bg=CARD)
        sf2.pack(pady=8)
        for num, lbl, page in steps:
            col_b = card(sf2)
            col_b.pack(side="left", padx=6)
            tk.Label(col_b, text=num, font=("Segoe UI", 18, "bold"),
                     fg=BTN_BG, bg=CARD, width=3).pack()
            tk.Label(col_b, text=lbl, font=FONT_SM, fg=MUTED, bg=CARD).pack()

        qf = tk.Frame(self, bg=BG)
        qf.pack(pady=10)
        quick = [
            ("🔬 Health Analysis", AnalysisPage), ("⚖ BMI Calc", BMIPage),
            ("🌸 Cycle Tracker", CycleTrackerPage), ("📅 Appointments", AppointmentPage),
            ("📂 History", HistoryPage), ("⭐ Feedback", FeedbackPage),
            ("📊 ML Metrics", MLMetricsPage),  # ← NEW quick link
        ]
        for i, (lbl, pg) in enumerate(quick):
            styled_btn(qf, lbl, lambda p=pg: self.master.show_frame(p),
                       width=18).grid(row=i//3, column=i%3, padx=8, pady=6)

        quotes = [
            "\"Your body is worthy of care and attention.\"",
            "\"Small steps lead to big health changes.\"",
            "\"Knowledge is the first medicine.\"",
        ]
        tk.Label(self, text=random.choice(quotes),
                 font=("Segoe UI", 9, "italic"), fg=MUTED, bg=BG, wraplength=600).pack(pady=10)

# ══════════════════════════════════════════════
#    PAGE – DETAILS
# ══════════════════════════════════════════════
class DetailsPage(tk.Frame):
    def __init__(self, master):
        super().__init__(master, bg=BG)

    def on_show(self):
        for w in self.winfo_children(): w.destroy()
        NavBar(self, self.master)

        prog = tk.Frame(self, bg=CARD2)
        prog.pack(fill="x")
        tk.Label(prog, text="STEP 2 of 5  –  Select Your Health Concern",
                 font=FONT_B, fg=ACCENT, bg=CARD2).pack(pady=8)
        bar = tk.Frame(prog, bg=BG, height=4)
        bar.pack(fill="x")
        tk.Frame(bar, bg=ACCENT, height=4, width=0).place(relwidth=0.4, relheight=1)

        inner = make_scrollable(self)
        c = card(inner)
        c.pack(padx=60, pady=12, fill="x")

        tk.Label(c, text="📋 Health Details",
                 font=FONT_T, fg=TEXT, bg=CARD).pack(pady=(16, 4))
        tk.Label(c, text="Select your primary health concern and provide details",
                 font=FONT_SM, fg=MUTED, bg=CARD).pack(pady=(0, 12))

        tk.Label(c, text="Your Age Group", font=FONT_B, fg=ACCENT2, bg=CARD).pack(pady=(8, 4))
        self.age_var = tk.StringVar(value="Adult (18–45)")
        age_groups = ["Child (under 12)", "Teen (12–18)", "Adult (18–45)",
                      "Middle Age (45–60)", "Senior (60+)"]
        ttk.Combobox(c, values=age_groups, textvariable=self.age_var,
                     state="readonly", width=35, font=FONT).pack(pady=4)

        tk.Label(c, text="Primary Health Concern", font=FONT_B, fg=ACCENT2, bg=CARD).pack(pady=(12, 4))
        self.prob_var = tk.StringVar()
        problems = list(PROBLEM_MAP.keys())
        style = ttk.Style(); style.theme_use("clam")
        style.configure("TCombobox", fieldbackground=ENTRY_BG,
                        background=ENTRY_BG, foreground=TEXT)
        self.combo = ttk.Combobox(c, values=problems, textvariable=self.prob_var,
                                  state="readonly", width=38, font=FONT)
        self.combo.pack(pady=4)

        tk.Label(c, text="How long have you had this concern?",
                 font=FONT_B, fg=ACCENT2, bg=CARD).pack(pady=(12, 4))
        self.dur_var = tk.StringVar(value="Less than 1 month")
        durations = ["Less than 1 month", "1–3 months", "3–6 months",
                     "6–12 months", "More than 1 year"]
        ttk.Combobox(c, values=durations, textvariable=self.dur_var,
                     state="readonly", width=35, font=FONT).pack(pady=4)

        tk.Label(c, text="Severity (1 = Mild, 5 = Severe)",
                 font=FONT_B, fg=ACCENT2, bg=CARD).pack(pady=(12, 4))
        self.severity = tk.IntVar(value=3)
        sev_f = tk.Frame(c, bg=CARD)
        sev_f.pack()
        for v in range(1, 6):
            tk.Radiobutton(sev_f, text=str(v), variable=self.severity, value=v,
                           bg=CARD, fg=TEXT, activebackground=CARD,
                           selectcolor=BTN_BG, font=FONT_B).pack(side="left", padx=8)

        tk.Label(c, text="Additional Symptoms / Notes",
                 font=FONT_B, fg=ACCENT2, bg=CARD).pack(pady=(12, 4))
        self.notes = tk.Text(c, height=4, bg=ENTRY_BG, fg=TEXT,
                             font=FONT, insertbackground=ACCENT, relief="flat")
        self.notes.pack(padx=20, pady=4, fill="x")

        btn_f = tk.Frame(c, bg=CARD)
        btn_f.pack(pady=16)
        styled_btn(btn_f, "Continue to Analysis  →", self.save, width=26).pack(side="left", padx=8)
        styled_btn(btn_f, "← Dashboard", lambda: self.master.show_frame(DashboardPage),
                   bg="#1e1040", width=14).pack(side="left", padx=4)

    def save(self):
        prob = self.prob_var.get()
        if not prob:
            messagebox.showwarning("Missing", "Please select a health concern.")
            return
        self.master.current_user.update({
            "problem":      prob,
            "age_group":    self.age_var.get(),
            "duration":     self.dur_var.get(),
            "severity":     self.severity.get(),
            "notes":        self.notes.get("1.0", tk.END).strip()
        })
        self.master.show_frame(AnalysisPage)

# ══════════════════════════════════════════════
#  PAGE – ANALYSIS
# ══════════════════════════════════════════════
class AnalysisPage(tk.Frame):
    def __init__(self, master):
        super().__init__(master, bg=BG)

    def on_show(self):
        for w in self.winfo_children(): w.destroy()
        NavBar(self, self.master)

        tk.Frame(self, bg=ACCENT, height=3).pack(fill="x")
        hdr = tk.Frame(self, bg=CARD2)
        hdr.pack(fill="x")
        tk.Label(hdr, text="🔬 AI Health Analysis",
                 font=FONT_T, fg=TEXT, bg=CARD2).pack(pady=(10, 2))
        tk.Label(hdr, text="STEP 3–4 of 5  –  AI-Generated Solutions",
                 font=FONT_SM, fg=MUTED, bg=CARD2).pack(pady=(0, 8))

        btn_row = tk.Frame(self, bg=BG)
        btn_row.pack(pady=8)
        styled_btn(btn_row, "⚡ Generate Analysis", self.generate, width=22).pack(side="left", padx=6)
        styled_btn(btn_row, "📄 Export PDF", self.export_pdf,
                   bg=BTN_OK, width=16).pack(side="left", padx=6)
        styled_btn(btn_row, "← Select Problem",
                   lambda: self.master.show_frame(DetailsPage),
                   bg="#1e1040", width=16).pack(side="left", padx=6)

        self.scroll_inner = make_scrollable(self, bg=BG)
        self._last_report = ""

        user = self.master.current_user
        if user.get("problem"):
            self.generate()

    def generate(self):
        for w in self.scroll_inner.winfo_children(): w.destroy()

        user    = self.master.current_user
        problem = user.get("problem", "")
        if not problem:
            tk.Label(self.scroll_inner,
                     text="⚠ Please go to Details page and select a health concern first.",
                     font=FONT_B, fg=WARNING, bg=BG).pack(pady=40)
            return

        detail_key = PROBLEM_MAP.get(problem, "")
        detail     = HEALTH_DETAILS.get(detail_key, {})
        h  = float(user.get("height") or 0)
        w  = float(user.get("weight") or 0)
        bmi, cat = bmi_calc(w, h) if h and w else (0, "N/A")
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

        pc = card(self.scroll_inner)
        pc.pack(fill="x", padx=20, pady=(12, 6))
        r1 = tk.Frame(pc, bg=CARD); r1.pack(fill="x", padx=18, pady=(14, 6))
        for lbl_txt, val in [
            ("Patient",    user.get("name", "N/A")),
            ("Age Group",  user.get("age_group", user.get("age", "N/A"))),
            ("Blood Group",user.get("blood_group", "N/A")),
            ("Date",       now[:10]),
            ("Severity",   f"{user.get('severity', 'N/A')}/5"),
            ("Duration",   user.get("duration", "N/A")),
        ]:
            col = tk.Frame(r1, bg=CARD)
            col.pack(side="left", expand=True)
            tk.Label(col, text=lbl_txt, font=FONT_SM, fg=MUTED, bg=CARD).pack()
            tk.Label(col, text=val, font=FONT_B, fg=TEXT, bg=CARD).pack()

        bmi_c = card(self.scroll_inner)
        bmi_c.pack(fill="x", padx=20, pady=4)
        tk.Label(bmi_c, text="⚖  Body Mass Index",
                 font=FONT_H, fg=ACCENT2, bg=CARD).pack(pady=(12, 4))
        col_map = {"Underweight": INFO, "Normal": SUCCESS, "Overweight": WARNING, "Obese": DANGER}
        tk.Label(bmi_c, text=f"{bmi}  –  {cat}",
                 font=("Segoe UI", 24, "bold"),
                 fg=col_map.get(cat, ACCENT), bg=CARD).pack()
        tk.Label(bmi_c, text="Healthy range: 18.5 – 24.9",
                 font=FONT_SM, fg=MUTED, bg=CARD).pack(pady=(0, 12))

        if not detail:
            tk.Label(self.scroll_inner,
                     text="No detailed data for this problem. Consult a doctor.",
                     font=FONT_B, fg=WARNING, bg=BG).pack(pady=30)
            return

        pc2 = card(self.scroll_inner)
        pc2.pack(fill="x", padx=20, pady=4)
        tk.Label(pc2, text=f"🩺  {problem}",
                 font=FONT_T, fg=ACCENT, bg=CARD).pack(pady=(14, 2))
        if detail.get("age_groups"):
            tk.Label(pc2, text=f"Commonly affects: {detail['age_groups']}",
                     font=FONT_SM, fg=MUTED, bg=CARD, wraplength=680).pack(pady=(0, 10))

        for section_name, items_key, icon, fg_col in [
            ("⚠  Common Causes",            "causes",      "▸", TEXT),
            ("🩺  Symptoms & Signs",         "symptoms",    "◦", ACCENT2),
            ("🛡  Precautions & Lifestyle",  "precautions", "✓", SUCCESS),
            ("🥗  Recommended Diet Plan",    "diet_plan",   "🍽", TEXT),
            ("🧘  Yoga & Exercise",          "yoga_poses",  "🌿", ACCENT2),
            ("💊  Supplements & Medicines",  "medicines",   "💊", WARNING),
            ("🏥  Recommended Specialists",  "doctors",     "👩‍⚕️", INFO),
        ]:
            if detail.get(items_key):
                s_card = card(self.scroll_inner)
                s_card.pack(fill="x", padx=20, pady=4)
                section_title(s_card, section_name)
                bullet_list(s_card, detail[items_key], icon, fg=fg_col)
                if items_key == "medicines":
                    tk.Label(s_card,
                             text="⚕ Always consult a doctor before starting any medication.",
                             font=("Segoe UI", 9, "italic"), fg=DANGER,
                             bg=CARD, wraplength=640).pack(pady=(6, 4), padx=18)
                tk.Frame(s_card, bg=BG, height=8).pack()

        nav_card = card(self.scroll_inner)
        nav_card.pack(fill="x", padx=20, pady=(8, 16))
        tk.Label(nav_card, text="✦  Analysis Complete!",
                 font=FONT_H, fg=SUCCESS, bg=CARD).pack(pady=(12, 6))
        btn_f2 = tk.Frame(nav_card, bg=CARD)
        btn_f2.pack(pady=(4, 14))
        styled_btn(btn_f2, "⭐ Go to Feedback  →",
                   lambda: self.master.show_frame(FeedbackPage),
                   bg=BTN_OK, width=22).pack(side="left", padx=6)
        styled_btn(btn_f2, "📅 Book Appointment",
                   lambda: self.master.show_frame(AppointmentPage),
                   width=20).pack(side="left", padx=6)
        styled_btn(btn_f2, "📊 ML Metrics",
                   lambda: self.master.show_frame(MLMetricsPage),
                   bg="#1e1040", width=14).pack(side="left", padx=6)

        lines = [f"AI WOMEN'S HEALTH ANALYSIS REPORT",
                 f"Patient: {user.get('name','N/A')}",
                 f"Concern: {problem}", f"Date: {now}",
                 f"BMI: {bmi} ({cat})", ""]
        for section, items in [
            ("CAUSES", detail.get("causes", [])),
            ("SYMPTOMS", detail.get("symptoms", [])),
            ("PRECAUTIONS", detail.get("precautions", [])),
            ("DIET PLAN", detail.get("diet_plan", [])),
            ("YOGA & EXERCISE", detail.get("yoga_poses", [])),
            ("MEDICINES", detail.get("medicines", [])),
            ("RECOMMENDED DOCTORS", detail.get("doctors", [])),
        ]:
            lines.append(f"\n{section}")
            lines.extend([f"  - {i}" for i in items])
        self._last_report = "\n".join(lines)

        cursor.execute(
            "INSERT INTO health_records(user_email,problem,solution,bmi,bmi_category,notes,recorded_at) VALUES(?,?,?,?,?,?,?)",
            (user.get("email"), problem, detail.get("medicines", [""])[0] or "See report",
             bmi, cat, user.get("notes", ""), now))
        conn.commit()

    def export_pdf(self):
        if not self._last_report:
            messagebox.showinfo("Info", "Generate analysis first.")
            return
        if not HAS_RL:
            messagebox.showwarning("Missing", "reportlab not installed. pip install reportlab")
            return
        path = filedialog.asksaveasfilename(defaultextension=".pdf",
                                            filetypes=[("PDF", "*.pdf")],
                                            initialfile="health_report.pdf")
        if not path: return
        try:
            doc    = SimpleDocTemplate(path, pagesize=letter,
                                       topMargin=0.5*inch, bottomMargin=0.5*inch)
            styles = getSampleStyleSheet()
            ts = ParagraphStyle("t", parent=styles["Title"], textColor=colors.purple, spaceAfter=12)
            bs = ParagraphStyle("b", parent=styles["Normal"], fontSize=10, spaceAfter=5)
            story = [Paragraph("AI Women's Health Analysis Report", ts), Spacer(1, 12)]
            for line in self._last_report.split("\n"):
                if line.strip():
                    story.append(Paragraph(line.strip(), bs))
            doc.build(story)
            messagebox.showinfo("Saved", f"PDF saved:\n{path}")
        except Exception as ex:
            messagebox.showerror("Error", str(ex))

# ══════════════════════════════════════════════
#               BMI PAGE
# ══════════════════════════════════════════════
class BMIPage(tk.Frame):
    def __init__(self, master):
        super().__init__(master, bg=BG)

    def on_show(self):
        for w in self.winfo_children(): w.destroy()
        NavBar(self, self.master)
        tk.Label(self, text="⚖  BMI Calculator", font=FONT_T, fg=TEXT, bg=BG).pack(pady=(18, 6))

        c = card(self)
        c.pack(padx=120, pady=8, fill="x")
        for lbl, attr, key in [("Weight (kg)", "bmi_w", "weight"), ("Height (cm)", "bmi_h", "height")]:
            tk.Label(c, text=lbl, font=FONT_B, fg=ACCENT, bg=CARD).pack(pady=(12, 2))
            e = entry(c)
            e.insert(0, str(self.master.current_user.get(key, "") or ""))
            setattr(self, attr, e)
            e.pack()
        styled_btn(c, "Calculate", self.calc).pack(pady=14)
        self.res_f = tk.Frame(c, bg=CARD); self.res_f.pack(fill="x", padx=20, pady=(0, 16))
        self.chart_f = tk.Frame(self, bg=BG); self.chart_f.pack(fill="both", expand=True, padx=30, pady=8)

    def calc(self):
        for w in self.res_f.winfo_children(): w.destroy()
        for w in self.chart_f.winfo_children(): w.destroy()
        try:
            weight = float(self.bmi_w.get()); height = float(self.bmi_h.get())
        except ValueError:
            messagebox.showerror("Error", "Enter valid numbers."); return
        bmi, cat = bmi_calc(weight, height)
        cols = {"Underweight": INFO, "Normal": SUCCESS, "Overweight": WARNING, "Obese": DANGER}
        col = cols.get(cat, ACCENT)
        tk.Label(self.res_f, text=f"BMI: {bmi}", font=FONT_T, fg=col, bg=CARD).pack(pady=(10, 2))
        tk.Label(self.res_f, text=cat, font=FONT_H, fg=col, bg=CARD).pack()
        tips = {
            "Underweight": "Increase caloric intake with nutritious foods & strength training.",
            "Normal":      "Excellent! Maintain your healthy lifestyle. 🎉",
            "Overweight":  "Add 30 min cardio daily & reduce processed foods.",
            "Obese":       "Consult a nutritionist and doctor for a structured plan."
        }
        tk.Label(self.res_f, text=tips.get(cat, ""), font=FONT, fg=TEXT,
                 bg=CARD, wraplength=400).pack(pady=6)
        if HAS_MPL:
            fig, ax = plt.subplots(figsize=(5, 2.2), facecolor=CARD); ax.set_facecolor(CARD)
            for lo, hi, c_, lbl in [(0,18.5,INFO,"Under"),(18.5,25,SUCCESS,"Normal"),(25,30,WARNING,"Over"),(30,40,DANGER,"Obese")]:
                ax.barh(0, hi-lo, left=lo, height=0.5, color=c_, alpha=0.9)
                ax.text((lo+hi)/2, 0, lbl, ha="center", va="center", fontsize=8, color="white", fontweight="bold")
            ax.axvline(min(bmi, 40), color="white", lw=3, ymin=0.05, ymax=0.95)
            ax.text(min(bmi,40), 0.32, f"  {bmi}", color="white", fontsize=10, fontweight="bold")
            ax.set_xlim(10, 40); ax.set_ylim(-0.6, 0.7); ax.axis("off")
            ax.set_title("BMI Scale", color=TEXT, fontsize=11, fontweight="bold")
            plt.tight_layout()
            cw = FigureCanvasTkAgg(fig, master=self.chart_f); cw.draw()
            cw.get_tk_widget().pack(fill="x"); plt.close(fig)

# ══════════════════════════════════════════════
#             CYCLE TRACKER
# ══════════════════════════════════════════════
class CycleTrackerPage(tk.Frame):
    def __init__(self, master):
        super().__init__(master, bg=BG)

    def on_show(self):
        for w in self.winfo_children(): w.destroy()
        NavBar(self, self.master)
        tk.Label(self, text="🌸  Menstrual Cycle Tracker",
                 font=FONT_T, fg=TEXT, bg=BG).pack(pady=(14, 4))
        main = tk.Frame(self, bg=BG); main.pack(fill="both", expand=True, padx=20)

        left = card(main); left.pack(side="left", fill="both", expand=True, padx=(0,8), pady=8)
        for lbl, attr, hint in [
            ("Period Start (YYYY-MM-DD)", "c_start", "2024-03-01"),
            ("Period End   (YYYY-MM-DD)", "c_end",   "2024-03-05"),
            ("Cycle Length (days)",        "c_len",   "28"),
            ("Notes",                      "c_notes", "Optional"),
        ]:
            tk.Label(left, text=lbl, font=FONT_B, fg=ACCENT, bg=CARD).pack(pady=(12,2))
            e = entry(left)
            e.insert(0, hint)
            e.bind("<FocusIn>", lambda ev, h=hint, en=e: en.delete(0, tk.END) if en.get()==h else None)
            setattr(self, attr, e); e.pack()
        styled_btn(left, "💾 Save Log", self.save).pack(pady=14)

        right = card(main); right.pack(side="left", fill="both", expand=True, padx=(8,0), pady=8)
        tk.Label(right, text="📊 Predictions", font=FONT_H, fg=ACCENT2, bg=CARD).pack(pady=(14,4))
        self.pred = tk.Text(right, height=14, bg=ENTRY_BG, fg=TEXT,
                            font=FONT_MONO, relief="flat", wrap="word")
        self.pred.pack(padx=10, pady=4, fill="both", expand=True)
        styled_btn(right, "🔄 Refresh", self.show_pred).pack(pady=(4,12))
        self.show_pred()

    def save(self):
        email = self.master.current_user.get("email","")
        try: length = int(self.c_len.get())
        except: length = 28
        cursor.execute("INSERT INTO cycle_tracker(user_email,period_start,period_end,cycle_length,notes) VALUES(?,?,?,?,?)",
                       (email, self.c_start.get(), self.c_end.get(), length, self.c_notes.get()))
        conn.commit(); messagebox.showinfo("Saved","Cycle log saved!"); self.show_pred()

    def show_pred(self):
        self.pred.delete("1.0", tk.END)
        email = self.master.current_user.get("email","")
        cursor.execute("SELECT * FROM cycle_tracker WHERE user_email=? ORDER BY id DESC LIMIT 5", (email,))
        rows = cursor.fetchall()
        if not rows:
            self.pred.insert(tk.END, "No data yet. Add your first log!"); return
        try:
            latest = rows[0]
            last   = datetime.datetime.strptime(latest[2], "%Y-%m-%d")
            cl     = latest[4] or 28
            nxt    = last + datetime.timedelta(days=cl)
            ov     = last + datetime.timedelta(days=cl-14)
            fs     = ov   - datetime.timedelta(days=2)
            fe     = ov   + datetime.timedelta(days=2)
            self.pred.insert(tk.END,
                f"🌸 CYCLE PREDICTIONS\n{'─'*32}\n"
                f"Last Start    : {latest[2]}\n"
                f"Last End      : {latest[3]}\n"
                f"Cycle Length  : {cl} days\n\n"
                f"📅 Next Period : {nxt.strftime('%Y-%m-%d')}\n"
                f"🥚 Ovulation   : {ov.strftime('%Y-%m-%d')}\n"
                f"💫 Fertile     : {fs.strftime('%d %b')} – {fe.strftime('%d %b')}\n\n"
                f"{'─'*32}\n📝 Recent Logs\n")
            for r in rows:
                self.pred.insert(tk.END, f"• {r[2]} → {r[3]} ({r[4]} days)\n")
        except Exception as ex:
            self.pred.insert(tk.END, f"Use YYYY-MM-DD format.\n{ex}")

# ══════════════════════════════════════════════
#             APPOINTMENTS
# ══════════════════════════════════════════════
class AppointmentPage(tk.Frame):
    def __init__(self, master):
        super().__init__(master, bg=BG)

    def on_show(self):
        for w in self.winfo_children(): w.destroy()
        NavBar(self, self.master)
        tk.Label(self, text="📅  Appointment Booking",
                 font=FONT_T, fg=TEXT, bg=BG).pack(pady=(14,4))
        main = tk.Frame(self, bg=BG); main.pack(fill="both", expand=True, padx=20)

        left = card(main); left.pack(side="left", fill="both", expand=True, padx=(0,8), pady=8)
        tk.Label(left, text="Doctor Name", font=FONT_B, fg=ACCENT, bg=CARD).pack(pady=(12,2))
        self.doc = entry(left); self.doc.pack()
        tk.Label(left, text="Specialty", font=FONT_B, fg=ACCENT, bg=CARD).pack(pady=(10,2))
        self.spec = tk.StringVar()
        ttk.Combobox(left,
            values=["Gynaecologist","Nutritionist","Dermatologist",
                    "Physiotherapist","Endocrinologist","Psychiatrist",
                    "General Physician","Neurologist","Oncologist"],
            textvariable=self.spec, state="readonly", width=27, font=FONT).pack(pady=4)
        for lbl, attr in [("Date (YYYY-MM-DD)","appt_d"),("Time (e.g. 10:30 AM)","appt_t"),("Notes","appt_n")]:
            tk.Label(left, text=lbl, font=FONT_B, fg=ACCENT, bg=CARD).pack(pady=(10,2))
            e = entry(left); e.pack(); setattr(self, attr, e)
        styled_btn(left, "📌 Book Appointment", self.book).pack(pady=14)

        right = card(main); right.pack(side="left", fill="both", expand=True, padx=(8,0), pady=8)
        tk.Label(right, text="Your Appointments", font=FONT_H, fg=ACCENT2, bg=CARD).pack(pady=(14,4))
        self.appt_list = tk.Text(right, height=16, bg=ENTRY_BG, fg=TEXT,
                                 font=FONT_MONO, relief="flat", wrap="word")
        self.appt_list.pack(padx=10, pady=4, fill="both", expand=True)
        self.refresh()

    def book(self):
        email = self.master.current_user.get("email","")
        cursor.execute(
            "INSERT INTO appointments(user_email,doctor_name,specialty,date,time,notes,status) VALUES(?,?,?,?,?,?,?)",
            (email, self.doc.get(), self.spec.get(), self.appt_d.get(), self.appt_t.get(), self.appt_n.get(), "Scheduled"))
        conn.commit(); messagebox.showinfo("Booked","Appointment booked!"); self.refresh()

    def refresh(self):
        self.appt_list.delete("1.0", tk.END)
        email = self.master.current_user.get("email","")
        cursor.execute("SELECT * FROM appointments WHERE user_email=? ORDER BY date DESC", (email,))
        rows = cursor.fetchall()
        if not rows:
            self.appt_list.insert(tk.END, "No appointments yet."); return
        for r in rows:
            self.appt_list.insert(tk.END,
                f"📅 {r[4]} {r[5]}\n   Dr. {r[2]} – {r[3]}\n   {r[6]}  [{r[7]}]\n{'─'*30}\n")

# ══════════════════════════════════════════════
#              HISTORY
# ══════════════════════════════════════════════
class HistoryPage(tk.Frame):
    def __init__(self, master):
        super().__init__(master, bg=BG)

    def on_show(self):
        for w in self.winfo_children(): w.destroy()
        NavBar(self, self.master)
        tk.Label(self, text="📂  Health History", font=FONT_T, fg=TEXT, bg=BG).pack(pady=(14,4))

        tab_ctrl = ttk.Notebook(self)
        tab_ctrl.pack(fill="both", expand=True, padx=16, pady=8)
        style = ttk.Style()
        style.configure("TNotebook", background=BG)
        style.configure("TNotebook.Tab", background=CARD, foreground=TEXT, padding=[12,6], font=FONT_B)

        email = self.master.current_user.get("email","")

        t1 = tk.Frame(tab_ctrl, bg=BG); tab_ctrl.add(t1, text="🩺 Records")
        cols = ("Date","Problem","BMI","Category")
        tree = ttk.Treeview(t1, columns=cols, show="headings", height=12)
        style.configure("Treeview", background=CARD, foreground=TEXT,
                        fieldbackground=CARD, rowheight=26, font=FONT)
        style.configure("Treeview.Heading", background=BTN_BG, foreground=TEXT, font=FONT_B)
        for c_ in cols: tree.heading(c_, text=c_); tree.column(c_, width=170)
        cursor.execute("SELECT recorded_at,problem,bmi,bmi_category FROM health_records WHERE user_email=? ORDER BY id DESC", (email,))
        for r in cursor.fetchall(): tree.insert("", tk.END, values=r)
        tree.pack(fill="both", expand=True, padx=10, pady=10)

        t2 = tk.Frame(tab_ctrl, bg=BG); tab_ctrl.add(t2, text="📊 BMI Trend")
        cursor.execute("SELECT recorded_at,bmi FROM health_records WHERE user_email=? AND bmi>0 ORDER BY id", (email,))
        rows = cursor.fetchall()
        if rows and HAS_MPL:
            dates = [r[0][:10] for r in rows]; bmis = [r[1] for r in rows]
            fig, ax = plt.subplots(figsize=(6,3), facecolor=BG); ax.set_facecolor(CARD)
            ax.plot(dates, bmis, color=ACCENT, lw=2, marker="o", ms=6)
            ax.axhline(18.5, color=INFO, ls="--", alpha=0.6, label="Underweight")
            ax.axhline(25, color=SUCCESS, ls="--", alpha=0.6, label="Normal upper")
            ax.set_title("BMI Over Time", color=TEXT, fontsize=12)
            ax.tick_params(colors=TEXT, labelrotation=20); ax.spines[:].set_color(MUTED)
            ax.legend(facecolor=CARD, labelcolor=TEXT, fontsize=8)
            plt.tight_layout()
            cw = FigureCanvasTkAgg(fig, master=t2); cw.draw()
            cw.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10); plt.close(fig)
        else:
            tk.Label(t2, text="No BMI data yet.", font=FONT, fg=MUTED, bg=BG).pack(pady=40)

        t3 = tk.Frame(tab_ctrl, bg=BG); tab_ctrl.add(t3, text="📥 Export")
        tk.Label(t3, text="Export health records to CSV", font=FONT_H, fg=ACCENT, bg=BG).pack(pady=30)
        styled_btn(t3, "⬇ Download CSV", lambda: self.export_csv(email), width=20).pack()

    def export_csv(self, email):
        cursor.execute("SELECT * FROM health_records WHERE user_email=?", (email,))
        rows = cursor.fetchall()
        if not HAS_PANDAS:
            messagebox.showwarning("Missing","pandas not installed."); return
        cols = ["id","email","problem","solution","bmi","bmi_cat","cycle_day","notes","date"]
        df = pd.DataFrame(rows, columns=cols)
        path = filedialog.asksaveasfilename(defaultextension=".csv",
                                            filetypes=[("CSV","*.csv")],
                                            initialfile="health_data.csv")
        if path: df.to_csv(path, index=False); messagebox.showinfo("Exported", f"Saved:\n{path}")

# ══════════════════════════════════════════════
#    PAGE 5 – FEEDBACK
# ══════════════════════════════════════════════
class FeedbackPage(tk.Frame):
    def __init__(self, master):
        super().__init__(master, bg=BG)

    def on_show(self):
        for w in self.winfo_children(): w.destroy()
        NavBar(self, self.master)

        tk.Frame(self, bg=ACCENT, height=3).pack(fill="x")
        hdr = tk.Frame(self, bg=CARD2); hdr.pack(fill="x")
        tk.Label(hdr, text="STEP 5 of 5  –  Feedback & Exit",
                 font=FONT_B, fg=ACCENT, bg=CARD2).pack(pady=8)

        dot_f = tk.Frame(hdr, bg=CARD2); dot_f.pack(pady=(0, 8))
        for i in range(1, 6):
            col = ACCENT if i == 5 else SUCCESS
            tk.Label(dot_f, text="●", font=("Segoe UI", 14), fg=col, bg=CARD2).pack(side="left", padx=3)

        inner = make_scrollable(self)

        c = card(inner)
        c.pack(padx=80, pady=(12, 6), fill="x")
        tk.Label(c, text="⭐  Share Your Feedback",
                 font=FONT_T, fg=TEXT, bg=CARD).pack(pady=(16, 4))
        tk.Label(c, text="Help us improve women's healthcare AI",
                 font=FONT_SM, fg=MUTED, bg=CARD).pack(pady=(0, 10))

        tk.Label(c, text="Your Experience", font=FONT_B, fg=ACCENT2, bg=CARD).pack(pady=(8, 4))
        self.fb = tk.Text(c, height=5, bg=ENTRY_BG, fg=TEXT,
                          font=FONT, relief="flat", insertbackground=ACCENT)
        self.fb.pack(padx=24, pady=4, fill="x")

        tk.Label(c, text="Category", font=FONT_B, fg=ACCENT2, bg=CARD).pack(pady=(8, 4))
        self.cat_var = tk.StringVar(value="General")
        cats = ["General", "Analysis Quality", "UI Design", "Suggestions", "Bug Report"]
        ttk.Combobox(c, values=cats, textvariable=self.cat_var,
                     state="readonly", width=28, font=FONT).pack(pady=4)

        tk.Label(c, text="Rating", font=FONT_B, fg=ACCENT2, bg=CARD).pack(pady=(10, 4))
        self.star_f  = tk.Frame(c, bg=CARD); self.star_f.pack()
        self.rating  = tk.IntVar(value=0)
        self.s_btns  = []
        for i in range(1, 6):
            b = tk.Button(self.star_f, text="☆", font=("Segoe UI", 22),
                          bg=CARD, fg=WARNING, relief="flat", cursor="hand2",
                          command=lambda v=i: self.set_star(v))
            b.pack(side="left", padx=3)
            self.s_btns.append(b)

        tk.Label(c, text="Would you recommend WellnessAI?",
                 font=FONT_B, fg=ACCENT2, bg=CARD).pack(pady=(10, 4))
        self.rec_var = tk.StringVar(value="Yes")
        rec_f = tk.Frame(c, bg=CARD); rec_f.pack()
        for opt in ["Yes", "Maybe", "No"]:
            tk.Radiobutton(rec_f, text=opt, variable=self.rec_var, value=opt,
                           bg=CARD, fg=TEXT, activebackground=CARD,
                           selectcolor=BTN_BG, font=FONT_B).pack(side="left", padx=10)

        styled_btn(c, "✅  Submit Feedback", self.save, bg=BTN_OK, width=24).pack(pady=(14, 0))

        s_card = card(inner)
        s_card.pack(padx=80, pady=6, fill="x")
        cursor.execute("SELECT AVG(rating), COUNT(*) FROM feedback")
        avg, cnt = cursor.fetchone()
        avg = round(avg or 0, 1)
        tk.Label(s_card, text="📊  Community Ratings",
                 font=FONT_H, fg=ACCENT2, bg=CARD).pack(pady=(12, 4))
        stars_shown = "⭐" * int(avg) + "☆" * (5 - int(avg))
        tk.Label(s_card, text=f"{stars_shown}  {avg}/5  |  {cnt} reviews",
                 font=FONT_B, fg=SUCCESS, bg=CARD).pack(pady=(0, 12))

        exit_card = card(inner)
        exit_card.pack(padx=80, pady=(6, 20), fill="x")
        tk.Frame(exit_card, bg=BORDER, height=1).pack(fill="x", padx=20, pady=(14, 10))
        tk.Label(exit_card, text="Thank you for using AI Women's Health Analyser",
                 font=FONT_H, fg=ACCENT, bg=CARD).pack()
        tk.Label(exit_card,
                 text="Your health journey matters. Come back anytime. 💜",
                 font=FONT, fg=MUTED, bg=CARD).pack(pady=4)

        exit_f = tk.Frame(exit_card, bg=CARD); exit_f.pack(pady=(10, 16))
        styled_btn(exit_f, "🏠  Dashboard",
                   lambda: self.master.show_frame(DashboardPage),
                   bg="#1e1040", width=16).pack(side="left", padx=8)
        styled_btn(exit_f, "🚪  Exit App",
                   self.exit_app, bg=DANGER, width=16).pack(side="left", padx=8)

    def set_star(self, v):
        self.rating.set(v)
        for i, b in enumerate(self.s_btns):
            b.config(text="★" if i < v else "☆")

    def save(self):
        rating = self.rating.get()
        text   = self.fb.get("1.0", tk.END).strip()
        if not text or rating == 0:
            messagebox.showwarning("Missing", "Write feedback and select a star rating.")
            return
        email = self.master.current_user.get("email", "")
        cursor.execute(
            "INSERT INTO feedback(user_email,text,rating,created_at) VALUES(?,?,?,?)",
            (email, text, rating, datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
        conn.commit()
        messagebox.showinfo("Thank You! 💜",
                            "Your feedback has been saved.\n\nWe appreciate your time!")
        self.fb.delete("1.0", tk.END)
        self.set_star(0)
        self.on_show()

    def exit_app(self):
        if messagebox.askyesno("Exit", "Are you sure you want to close the app?"):
            conn.close()
            self.master.destroy()
            sys.exit(0)

# ══════════════════════════════════════════════
#   ★ NEW PAGE – ML METRICS (Accuracy, F1, etc.)
# ══════════════════════════════════════════════
class MLMetricsPage(tk.Frame):
    """
    Displays ML model evaluation metrics:
      • Accuracy, Precision, Recall, F1 Score (with gauge bars)
      • Full per-class classification report
      • Confusion matrix heatmap (matplotlib)
      • Algorithm details and interpretation guide
    """
    def __init__(self, master):
        super().__init__(master, bg=BG)

    def on_show(self):
        for w in self.winfo_children(): w.destroy()
        NavBar(self, self.master)

        # ── Header ──
        tk.Frame(self, bg=ACCENT, height=3).pack(fill="x")
        hdr = tk.Frame(self, bg=CARD2); hdr.pack(fill="x")
        tk.Label(hdr, text="📊  ML Model Metrics",
                 font=FONT_T, fg=TEXT, bg=CARD2).pack(pady=(10, 2))
        tk.Label(hdr, text="Algorithm performance evaluation  –  Multinomial Naive Bayes + TF-IDF",
                 font=FONT_SM, fg=MUTED, bg=CARD2).pack(pady=(0, 8))

        if not HAS_SK:
            no_sk = card(self); no_sk.pack(padx=60, pady=30, fill="x")
            tk.Label(no_sk, text="⚠  scikit-learn not installed",
                     font=FONT_H, fg=WARNING, bg=CARD).pack(pady=(20, 4))
            tk.Label(no_sk,
                     text="Install it with:  pip install scikit-learn numpy\nThen restart the app.",
                     font=FONT, fg=TEXT, bg=CARD).pack(pady=(4, 20))
            return

        if "error" in _ML_METRICS:
            err_c = card(self); err_c.pack(padx=60, pady=30, fill="x")
            tk.Label(err_c, text=f"Metrics error: {_ML_METRICS['error']}",
                     font=FONT, fg=DANGER, bg=CARD).pack(pady=20)
            return

        # ── Action buttons ──
        btn_row = tk.Frame(self, bg=BG); btn_row.pack(pady=6)
        styled_btn(btn_row, "🔄 Recompute Metrics", self.on_show, width=22).pack(side="left", padx=6)
        styled_btn(btn_row, "💾 Export Report", self._export_report, bg=BTN_OK, width=18).pack(side="left", padx=6)
        styled_btn(btn_row, "← Dashboard",
                   lambda: self.master.show_frame(DashboardPage),
                   bg="#1e1040", width=14).pack(side="left", padx=6)

        inner = make_scrollable(self, bg=BG)

        # ═══════════════════════════════════════
        # 1. ALGORITHM INFO CARD
        # ═══════════════════════════════════════
        algo_card = card(inner)
        algo_card.pack(fill="x", padx=20, pady=(12, 6))
        section_title(algo_card, "🤖  Algorithm Information")
        info_rows = [
            ("Algorithm",      _ML_METRICS.get("algo", "N/A")),
            ("Dataset Size",   f"{_ML_METRICS.get('n_samples', 0)} samples"),
            ("Classes",        f"{_ML_METRICS.get('n_classes', 0)} health conditions"),
            ("CV Strategy",    f"{_ML_METRICS.get('cv_folds', 0)}-Fold Stratified Cross-Validation"),
            ("Feature Extraction", "TF-IDF Vectorizer  (unigrams + bigrams)"),
            ("Evaluation",     "Cross-val predictions vs ground-truth labels"),
        ]
        grid_f = tk.Frame(algo_card, bg=CARD)
        grid_f.pack(fill="x", padx=24, pady=(4, 14))
        for i, (k, v) in enumerate(info_rows):
            row_bg = CARD if i % 2 == 0 else CARD2
            row = tk.Frame(grid_f, bg=row_bg)
            row.pack(fill="x")
            tk.Label(row, text=k, font=FONT_B, fg=ACCENT2, bg=row_bg,
                     width=26, anchor="w").pack(side="left", padx=10, pady=4)
            tk.Label(row, text=v, font=FONT, fg=TEXT, bg=row_bg,
                     anchor="w").pack(side="left", padx=4, pady=4)

        # ═══════════════════════════════════════
        # 2. KEY METRICS – GAUGE BARS
        # ═══════════════════════════════════════
        kpi_card = card(inner)
        kpi_card.pack(fill="x", padx=20, pady=6)
        section_title(kpi_card, "📈  Key Performance Metrics")

        metric_defs = [
            ("Accuracy",  _ML_METRICS.get("accuracy",  0),
             "Overall correct predictions out of all predictions made.",
             SUCCESS),
            ("Precision", _ML_METRICS.get("precision", 0),
             "Of all predicted positives, how many were actually positive (weighted avg).",
             INFO),
            ("Recall",    _ML_METRICS.get("recall",    0),
             "Of all actual positives, how many did the model correctly identify (weighted avg).",
             ACCENT),
            ("F1 Score",  _ML_METRICS.get("f1",        0),
             "Harmonic mean of Precision and Recall – balanced measure of model performance.",
             WARNING),
        ]

        for name, val, desc, col in metric_defs:
            row_f = tk.Frame(kpi_card, bg=CARD)
            row_f.pack(fill="x", padx=24, pady=6)

            # Left: label + value
            left_f = tk.Frame(row_f, bg=CARD, width=160)
            left_f.pack(side="left")
            left_f.pack_propagate(False)
            tk.Label(left_f, text=name, font=FONT_B, fg=col, bg=CARD).pack(anchor="w")
            tk.Label(left_f, text=f"{val}%", font=("Segoe UI", 16, "bold"),
                     fg=TEXT, bg=CARD).pack(anchor="w")

            # Middle: bar
            mid_f = tk.Frame(row_f, bg=CARD)
            mid_f.pack(side="left", fill="x", expand=True, padx=(8, 12))
            bar_bg = tk.Frame(mid_f, bg=BORDER, height=18)
            bar_bg.pack(fill="x", pady=6)
            fill_w = max(1, int(val))  # percentage width
            tk.Frame(bar_bg, bg=col, height=18,
                     width=0).place(relwidth=fill_w/100, relheight=1)

            # Right: description
            tk.Label(row_f, text=desc, font=FONT_SM, fg=MUTED, bg=CARD,
                     wraplength=300, justify="left").pack(side="left", padx=4)

        # Grade badge
        acc = _ML_METRICS.get("accuracy", 0)
        if acc >= 90:   grade, gcol = "A  –  Excellent",  SUCCESS
        elif acc >= 75: grade, gcol = "B  –  Good",       INFO
        elif acc >= 60: grade, gcol = "C  –  Moderate",   WARNING
        else:           grade, gcol = "D  –  Needs Work",  DANGER

        grade_f = tk.Frame(kpi_card, bg=CARD)
        grade_f.pack(pady=(4, 14))
        tk.Label(grade_f, text="Overall Grade:  ", font=FONT_B, fg=MUTED, bg=CARD).pack(side="left")
        tk.Label(grade_f, text=grade, font=("Segoe UI", 13, "bold"), fg=gcol, bg=CARD).pack(side="left")

        # ═══════════════════════════════════════
        # 3. CONFUSION MATRIX
        # ═══════════════════════════════════════
        cm_card = card(inner)
        cm_card.pack(fill="x", padx=20, pady=6)
        section_title(cm_card, "🔲  Confusion Matrix")
        tk.Label(cm_card,
                 text="Rows = Actual class  |  Columns = Predicted class  |  Diagonal = Correct predictions",
                 font=FONT_SM, fg=MUTED, bg=CARD).pack(padx=24, pady=(0, 4))

        if HAS_MPL and "cm" in _ML_METRICS:
            try:
                cm     = _ML_METRICS["cm"]
                labels = _ML_METRICS["labels"]
                n      = len(labels)

                # Dynamic figure size
                fig_w = max(8, n * 0.55)
                fig_h = max(6, n * 0.5)
                fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=CARD)
                ax.set_facecolor(CARD)

                # Normalize for color intensity (0–1)
                cm_norm = cm.astype(float)
                row_sums = cm_norm.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1
                cm_norm = cm_norm / row_sums

                cmap = plt.cm.RdPu
                im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1, aspect="auto")

                # Colour bar
                cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
                cbar.ax.yaxis.set_tick_params(color=TEXT)
                plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT, fontsize=8)

                # Axis labels
                ax.set_xticks(range(n)); ax.set_yticks(range(n))
                ax.set_xticklabels(labels, rotation=45, ha="right",
                                   fontsize=7.5, color=TEXT)
                ax.set_yticklabels(labels, fontsize=7.5, color=TEXT)
                ax.set_xlabel("Predicted Label", color=TEXT, fontsize=9, labelpad=8)
                ax.set_ylabel("Actual Label",    color=TEXT, fontsize=9, labelpad=8)
                ax.set_title("Confusion Matrix  (row-normalised)",
                             color=TEXT, fontsize=11, fontweight="bold", pad=10)
                ax.spines[:].set_color(MUTED)

                # Annotate cells
                for i in range(n):
                    for j in range(n):
                        raw_val  = cm[i, j]
                        norm_val = cm_norm[i, j]
                        txt_col  = "white" if norm_val > 0.5 else "#ccaaff"
                        if raw_val > 0:
                            ax.text(j, i, str(raw_val), ha="center", va="center",
                                    fontsize=7, color=txt_col, fontweight="bold")

                plt.tight_layout()
                chart_holder = tk.Frame(cm_card, bg=CARD)
                chart_holder.pack(fill="x", padx=20, pady=8)
                cw = FigureCanvasTkAgg(fig, master=chart_holder)
                cw.draw()
                cw.get_tk_widget().pack(fill="x")
                plt.close(fig)

            except Exception as e_cm:
                tk.Label(cm_card, text=f"Could not render confusion matrix: {e_cm}",
                         font=FONT_SM, fg=DANGER, bg=CARD).pack(pady=10)
        else:
            tk.Label(cm_card,
                     text="Install matplotlib to view the confusion matrix.",
                     font=FONT, fg=MUTED, bg=CARD).pack(pady=12)

        # ═══════════════════════════════════════
        # 4. CLASSIFICATION REPORT
        # ═══════════════════════════════════════
        rep_card = card(inner)
        rep_card.pack(fill="x", padx=20, pady=6)
        section_title(rep_card, "📝  Per-Class Classification Report")
        tk.Label(rep_card,
                 text="Precision, Recall, F1 and Support for each of the 26 health classes:",
                 font=FONT_SM, fg=MUTED, bg=CARD).pack(padx=24, pady=(0, 6))

        rep_txt = tk.Text(rep_card, height=22, bg=ENTRY_BG, fg=TEXT,
                          font=FONT_MONO, relief="flat", wrap="none")
        rep_txt.pack(padx=20, pady=(0, 6), fill="x")
        rep_scroll = ttk.Scrollbar(rep_card, orient="horizontal", command=rep_txt.xview)
        rep_txt.configure(xscrollcommand=rep_scroll.set)
        rep_scroll.pack(fill="x", padx=20)

        report_str = _ML_METRICS.get("report", "No report available.")
        rep_txt.insert(tk.END, report_str)
        rep_txt.config(state="disabled")

        tk.Frame(rep_card, bg=BG, height=8).pack()

        # ═══════════════════════════════════════
        # 5. INTERPRETATION GUIDE
        # ═══════════════════════════════════════
        guide_card = card(inner)
        guide_card.pack(fill="x", padx=20, pady=6)
        section_title(guide_card, "📖  How to Interpret These Metrics")

        guide_items = [
            ("Accuracy",
             "% of all test samples correctly classified. Simple and intuitive. Can be misleading on imbalanced datasets."),
            ("Precision",
             "When the model predicts class X, how often is it actually X? High precision = fewer false positives."),
            ("Recall",
             "Of all real class-X samples, how many did the model catch? High recall = fewer false negatives."),
            ("F1 Score",
             "Harmonic mean of precision and recall. Use when you want a single balanced metric. F1 = 2 × (P×R)/(P+R)."),
            ("Confusion Matrix",
             "Each cell (i, j) shows how many times actual class i was predicted as class j. Perfect model = diagonal only."),
            ("Weighted Average",
             "Each class metric is weighted by its support (number of true samples). Accounts for class imbalance."),
            ("Cross-Validation",
             "Data is split into N folds. The model trains on N-1 folds and tests on 1. Repeated N times. More reliable than a single split."),
        ]

        for term, explanation in guide_items:
            g_row = tk.Frame(guide_card, bg=CARD)
            g_row.pack(fill="x", padx=24, pady=3)
            tk.Label(g_row, text=f"  {term}:", font=FONT_B, fg=ACCENT2, bg=CARD,
                     width=18, anchor="w").pack(side="left")
            tk.Label(g_row, text=explanation, font=FONT_SM, fg=TEXT, bg=CARD,
                     anchor="w", wraplength=560, justify="left").pack(side="left", fill="x")

        tk.Frame(guide_card, bg=BG, height=12).pack()

        # ═══════════════════════════════════════
        # 6. BAR CHART – CLASS-LEVEL F1
        # ═══════════════════════════════════════
        if HAS_MPL and "cm" in _ML_METRICS:
            bar_card = card(inner)
            bar_card.pack(fill="x", padx=20, pady=(6, 16))
            section_title(bar_card, "📊  Per-Class F1 Score Distribution")
            tk.Label(bar_card,
                     text="Green bar = model identifies this class well  |  Red bar = room for improvement",
                     font=FONT_SM, fg=MUTED, bg=CARD).pack(padx=24, pady=(0, 4))
            try:
                labels  = _ML_METRICS["labels"]
                cm      = _ML_METRICS["cm"]
                n       = len(labels)
                f1_per  = []
                for i in range(n):
                    tp = cm[i, i]
                    fp = cm[:, i].sum() - tp
                    fn = cm[i, :].sum() - tp
                    denom = 2*tp + fp + fn
                    f1_per.append(round(2*tp / denom, 3) if denom else 0.0)

                colors_bar = [SUCCESS if v >= 0.7 else (WARNING if v >= 0.4 else DANGER)
                              for v in f1_per]
                fig2, ax2 = plt.subplots(figsize=(max(8, n*0.52), 3.5), facecolor=CARD)
                ax2.set_facecolor(CARD)
                bars = ax2.bar(range(n), f1_per, color=colors_bar, edgecolor=BORDER, linewidth=0.5)
                ax2.set_xticks(range(n))
                ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.5, color=TEXT)
                ax2.set_ylim(0, 1.1)
                ax2.set_ylabel("F1 Score", color=TEXT, fontsize=9)
                ax2.set_title("F1 Score per Class", color=TEXT, fontsize=11, fontweight="bold")
                ax2.tick_params(colors=TEXT)
                ax2.spines[:].set_color(MUTED)
                ax2.axhline(0.7, color=SUCCESS, lw=1, ls="--", alpha=0.5, label="Good threshold (0.7)")
                ax2.legend(facecolor=CARD, labelcolor=TEXT, fontsize=8)

                # Value labels on bars
                for bar, v in zip(bars, f1_per):
                    ax2.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                             f"{v:.2f}", ha="center", va="bottom",
                             fontsize=7, color=TEXT, fontweight="bold")

                plt.tight_layout()
                bh = tk.Frame(bar_card, bg=CARD)
                bh.pack(fill="x", padx=20, pady=8)
                cw2 = FigureCanvasTkAgg(fig2, master=bh)
                cw2.draw()
                cw2.get_tk_widget().pack(fill="x")
                plt.close(fig2)

            except Exception as e_bar:
                tk.Label(bar_card, text=f"Bar chart error: {e_bar}",
                         font=FONT_SM, fg=DANGER, bg=CARD).pack(pady=6)

    # ── Export metrics report ──
    def _export_report(self):
        if not HAS_SK or "error" in _ML_METRICS:
            messagebox.showwarning("Not available", "Metrics not computed yet.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text File", "*.txt"), ("All files", "*.*")],
            initialfile="ml_metrics_report.txt"
        )
        if not path: return
        try:
            lines = [
                "=" * 60,
                "  AI WOMEN'S HEALTH – ML MODEL METRICS REPORT",
                f"  Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
                "=" * 60, "",
                f"Algorithm      : {_ML_METRICS.get('algo', 'N/A')}",
                f"Dataset Size   : {_ML_METRICS.get('n_samples', 0)} samples",
                f"Classes        : {_ML_METRICS.get('n_classes', 0)} health conditions",
                f"CV Folds       : {_ML_METRICS.get('cv_folds', 0)}-Fold Stratified CV",
                "",
                "── KEY METRICS ──────────────────────────────────────",
                f"  Accuracy   : {_ML_METRICS.get('accuracy',  0)}%",
                f"  Precision  : {_ML_METRICS.get('precision', 0)}%  (weighted avg)",
                f"  Recall     : {_ML_METRICS.get('recall',    0)}%  (weighted avg)",
                f"  F1 Score   : {_ML_METRICS.get('f1',        0)}%  (weighted avg)",
                "",
                "── CLASSIFICATION REPORT ────────────────────────────",
                _ML_METRICS.get("report", ""),
                "",
                "── CONFUSION MATRIX (raw counts) ────────────────────",
            ]
            labels = _ML_METRICS.get("labels", [])
            cm     = _ML_METRICS.get("cm", [])
            col_w  = max(12, max(len(l) for l in labels) + 2) if labels else 14
            header = " " * col_w + "  ".join(f"{l:>{col_w}}" for l in labels)
            lines.append(header)
            for i, lbl in enumerate(labels):
                row_vals = "  ".join(f"{int(cm[i, j]):>{col_w}}" for j in range(len(labels)))
                lines.append(f"{lbl:<{col_w}}{row_vals}")
            lines += ["", "=" * 60,
                      "Note: Metrics are computed using cross-validation predictions.",
                      "These reflect the model's generalisation ability."]

            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            messagebox.showinfo("Exported ✅", f"ML metrics report saved:\n{path}")
        except Exception as ex:
            messagebox.showerror("Export Error", str(ex))


# ══════════════════════════════════════════════
#                    RUN
# ══════════════════════════════════════════════
if __name__ == "__main__":
    app = App()
    app.mainloop()
