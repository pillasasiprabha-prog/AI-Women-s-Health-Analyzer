from flask import Flask, render_template_string, request, redirect, url_for, session, jsonify, send_file
import sqlite3
import hashlib
import datetime
import random
import io
import os

# ── optional deps ──
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

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

app = Flask(__name__)
app.secret_key = "wellnessai_secret_2024"

# ══════════════════════════════════════════════
#                   DATABASE
# ══════════════════════════════════════════════
def get_db():
    conn = sqlite3.connect("womens_health_v2.db")
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.executescript("""
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
    conn.close()

init_db()

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

SHORT_LABEL_MAP = {
    "Meditation, Yoga, Therapy, Journaling, Breathing exercises": "Stress/Anxiety",
    "Low-calorie diet, 45min cardio daily, Reduce sugar/processed food": "Wt Gain",
    "High-protein diet, strength training, calorie surplus, frequent meals": "Wt Loss",
    "Hormone therapy, low-GI diet, regular exercise, stress reduction": "PCOS",
    "Gynecologist checkup, iron-rich diet, track cycle, avoid stress": "Irreg Period",
    "Prenatal vitamins, regular OB-GYN visits, balanced nutrition, rest": "Pregnancy",
    "Biotin supplements, scalp massage, protein-rich diet, gentle hair care": "Hair",
    "Hydration, SPF, Vitamin C serum, gentle cleanser, balanced diet": "Skin/Acne",
    "Physiotherapy, hot/cold therapy, anti-inflammatory diet, rest": "Body Pain",
    "Thyroid medication, iodine-rich diet, regular TSH tests": "Thyroid",
    "Low-GI diet, regular monitoring, exercise, medication if needed": "Diabetes",
    "Iron & B12 supplements, spinach, legumes, red meat (if non-veg)": "Anaemia",
    "Dark room rest, hydration, magnesium supplement, avoid triggers": "Migraine",
    "Sleep hygiene, no screens before bed, chamomile tea, melatonin": "Insomnia",
    "Core strengthening, ergonomic posture, physio, hot compress": "Back Pain",
    "Therapy/counselling, social support, exercise, medication if needed": "Depression",
    "Low-sodium diet, regular BP monitoring, aerobic exercise, medication": "Hypert.",
    "Probiotics, fibre-rich diet, hydration, reduce dairy/gluten": "Digestive",
    "Monthly self-exam, annual mammogram, healthy weight, no smoking": "Breast",
    "Calcium + Vitamin D, weight-bearing exercise, bone density scan": "Osteo",
    "HRT if needed, cooling techniques, soy isoflavones, yoga": "Menopause",
    "Hydration, D-Mannose, cranberry extract, antibiotics if severe": "UTI",
    "Pain management, hormonal therapy, laparoscopy, pelvic physio": "Endometrio.",
    "Therapy, partner support, medication, rest, gentle walks": "Postpartum",
    "Calcium-rich foods, balanced macros, limit junk, regular activity": "Child Nutr.",
    "Iron-rich foods, calcium, hygiene education, body positivity": "Teen Health",
}

_ML_METRICS = {}

if HAS_SK:
    import pandas as _pd
    df_health = _pd.DataFrame(health_data)
    X = df_health["problem"].tolist()
    y = df_health["solution"].tolist()
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("clf", MultinomialNB(alpha=0.5))
    ])
    pipeline.fit(X, y)
    try:
        unique_classes = list(dict.fromkeys(y))
        n_classes = len(unique_classes)
        n_splits = min(5, len(X) // n_classes) if len(X) // n_classes >= 2 else 2
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        y_pred_cv = cross_val_predict(pipeline, X, y, cv=cv)
        _ML_METRICS["accuracy"] = round(accuracy_score(y, y_pred_cv) * 100, 2)
        _ML_METRICS["precision"] = round(precision_score(y, y_pred_cv, average="weighted", zero_division=0) * 100, 2)
        _ML_METRICS["recall"] = round(recall_score(y, y_pred_cv, average="weighted", zero_division=0) * 100, 2)
        _ML_METRICS["f1"] = round(f1_score(y, y_pred_cv, average="weighted", zero_division=0) * 100, 2)
        _ML_METRICS["cm"] = confusion_matrix(y, y_pred_cv, labels=unique_classes).tolist()
        _ML_METRICS["labels"] = [SHORT_LABEL_MAP.get(c, c[:12]) for c in unique_classes]
        _ML_METRICS["full_labels"] = unique_classes
        _ML_METRICS["report"] = classification_report(y, y_pred_cv, target_names=_ML_METRICS["labels"], zero_division=0)
        _ML_METRICS["n_samples"] = len(X)
        _ML_METRICS["n_classes"] = n_classes
        _ML_METRICS["algo"] = "Multinomial Naive Bayes + TF-IDF (1-2 grams)"
        _ML_METRICS["cv_folds"] = n_splits
    except Exception as e:
        _ML_METRICS["error"] = str(e)

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
        "causes": ["Work/academic pressure","Relationship conflicts","Financial concerns","Hormonal imbalances (PMS, menopause)","Trauma or past abuse","Social media overexposure"],
        "symptoms": ["Constant worry or overthinking","Rapid heartbeat, sweating","Headaches and muscle tension","Fatigue and poor concentration","Irritability, mood swings","Sleep disturbances"],
        "precautions": ["Limit news/social media to 30 min/day","Practice box breathing (4-4-4-4 counts)","Set healthy boundaries at work and home","Keep a gratitude journal nightly","Talk to a trusted friend or therapist"],
        "diet_plan": ["🍌 Bananas & dark chocolate – mood boosters (serotonin)","🫐 Blueberries – antioxidants fight stress hormones","🥑 Avocado – B vitamins for nervous system","🌿 Green tea – L-theanine for calm focus","🐟 Fatty fish (salmon) – omega-3 reduces cortisol","🥜 Almonds & walnuts – magnesium & healthy fats","❌ Avoid caffeine, alcohol, excessive sugar"],
        "yoga_poses": ["🧘 Child's Pose (Balasana) – 5 min","🧘 Legs-Up-The-Wall (Viparita Karani) – 10 min","🧘 Cat-Cow stretch – 3 min","🧘 Corpse Pose (Savasana) – 10 min","🧘 Alternate Nostril Breathing (Nadi Shodhana)"],
        "medicines": ["Ashwagandha (adaptogen supplement)","Magnesium Glycinate 300mg","B-Complex vitamins","⚕ Prescribed: SSRIs or anxiolytics (consult psychiatrist)"],
        "doctors": ["Psychiatrist / Psychologist","Gynaecologist (if hormonal)","General Physician"]
    },
    "weight gain / obesity": {
        "age_groups": "All ages; post-pregnancy, menopause, PCOD-related",
        "causes": ["Sedentary lifestyle","Hormonal imbalances (thyroid, PCOD)","Emotional eating","Sleep deprivation","Medications (steroids, antidepressants)","Insulin resistance"],
        "symptoms": ["BMI above 25 (overweight) or 30 (obese)","Fatigue with minimal activity","Joint pain (knees, hips)","Snoring / sleep apnoea","Irregular periods","Low self-esteem"],
        "precautions": ["Track calories with a food diary (target 500 kcal deficit)","Never skip breakfast – fuels metabolism","Drink 500 ml water before each meal","Eat slowly; chew 20–30 times per bite","Sleep 7–8 hrs; sleep deprivation increases ghrelin"],
        "diet_plan": ["🥗 Plate rule: ½ veggies, ¼ protein, ¼ complex carbs","🍗 Lean proteins: chicken, tofu, lentils, eggs","🥦 Non-starchy veggies: broccoli, spinach, zucchini","🍓 Low-sugar fruits: berries, apple, pear","🌾 Complex carbs: oats, quinoa, brown rice","❌ Cut: sugary drinks, fried snacks, white bread, alcohol"],
        "yoga_poses": ["🔥 Sun Salutation (Surya Namaskar) – 12 rounds/day","🔥 Warrior I & II – tones legs & core","🔥 Boat Pose (Navasana) – core strength","🔥 Bridge Pose – activates glutes & metabolism","🔥 Twisted Chair Pose – detox & toning"],
        "medicines": ["Vitamin D3 + K2 (often deficient in obesity)","Omega-3 fish oil 1000mg","Probiotics for gut health","⚕ Prescribed: Orlistat / Metformin (if insulin resistant)"],
        "doctors": ["Nutritionist / Dietitian","Endocrinologist","Bariatric Specialist (if BMI > 35)"]
    },
    "weight loss / underweight": {
        "age_groups": "Teens, young adults, post-illness recovery",
        "causes": ["Inadequate caloric intake","Hyperthyroidism","Eating disorders (anorexia, bulimia)","Malabsorption / IBS / Crohn's disease","Depression or chronic stress","Parasitic infections"],
        "symptoms": ["BMI below 18.5","Fatigue and weakness","Brittle nails and hair loss","Frequent illness (low immunity)","Irregular or absent periods","Poor wound healing"],
        "precautions": ["Eat every 3–4 hours; never skip meals","Add healthy calorie-dense snacks between meals","Avoid excessive cardio – focus on strength training","Rule out thyroid / digestive disorders with blood tests","Address emotional causes with a therapist if needed"],
        "diet_plan": ["🥛 Full-fat dairy: milk, Greek yoghurt, paneer","🥜 Nuts & nut butters: almonds, peanut butter, cashews","🍌 High-calorie fruits: banana, mango, avocado, dates","🍚 Complex carbs: rice, whole wheat roti, sweet potato","🥚 Eggs + legumes – protein for muscle gain","🫒 Healthy oils: olive oil, ghee in cooking","✅ Aim: 300–500 kcal surplus above daily needs"],
        "yoga_poses": ["💪 Warrior III – builds muscle & balance","💪 Chair Pose (Utkatasana) – lower body strength","💪 Cobra Pose – spinal strength, digestion","💪 Shoulder Stand (Sarvangasana) – thyroid stimulation","💪 Downward Dog – full-body toning"],
        "medicines": ["Protein supplements (whey / plant-based) post-workout","Multivitamin with zinc & iron","Vitamin B12 (especially if vegetarian)","⚕ If thyroid: consult endocrinologist for medication"],
        "doctors": ["Nutritionist / Dietitian","Endocrinologist","Gastroenterologist","Psychiatrist (if eating disorder)"]
    },
    "pcod / pcos": {
        "age_groups": "Teens to women aged 12–45",
        "causes": ["Insulin resistance (most common cause)","Elevated androgens (male hormones)","Genetic predisposition","Chronic inflammation","Sedentary lifestyle + poor diet","Stress and sleep deprivation"],
        "symptoms": ["Irregular or absent periods","Excess facial / body hair (hirsutism)","Acne and oily skin","Weight gain (especially belly fat)","Hair thinning on scalp","Difficulty conceiving","Dark skin patches (acanthosis nigricans)","Mood swings"],
        "precautions": ["Lose even 5–10% body weight – restores periods in many women","Avoid refined sugar and white carbohydrates completely","Practise seed cycling (flax+pumpkin days 1–14; sesame+sunflower days 15–28)","Check fasting insulin, testosterone, AMH every 6 months","Use non-comedogenic skincare products"],
        "diet_plan": ["🌾 Low-GI foods: oats, quinoa, millet, barley","🥦 Anti-inflammatory veggies: broccoli, kale, spinach","🍒 Berries + cherries – lower insulin spikes","🐟 Fatty fish 3x/week – reduce inflammation","🫘 Lentils, chickpeas – plant protein + fibre","🌰 Spearmint tea – reduces androgen levels naturally","❌ Avoid: dairy excess, white sugar, processed carbs, soy"],
        "yoga_poses": ["🌸 Butterfly Pose (Baddha Konasana) – opens hips","🌸 Reclined Butterfly – relaxes pelvic organs","🌸 Garland Pose (Malasana) – improves circulation","🌸 Supported Bridge – balances hormones","🌸 Supta Matsyendrasana – detoxes abdominal organs"],
        "medicines": ["Myo-Inositol 2000mg + D-Chiro-Inositol 50mg (evidence-based)","Vitamin D3 2000–4000 IU (most PCOS women are deficient)","Omega-3 fatty acids 1–2g/day","Spearmint capsules / tea (anti-androgenic)","⚕ Prescribed: Metformin, OCP, Clomiphene (for fertility)"],
        "doctors": ["Gynaecologist / Reproductive Endocrinologist","Endocrinologist","Nutritionist","Dermatologist (for skin/hair)"]
    },
    "pregnancy / prenatal care": {
        "age_groups": "Women aged 18–45 (pregnancy and postpartum)",
        "causes": ["Normal physiological process; complications from nutrition, stress, infections"],
        "symptoms": ["1st Trimester: nausea, fatigue, breast tenderness, frequent urination","2nd Trimester: visible bump, fetal movements, back pain","3rd Trimester: heartburn, breathlessness, swollen feet, Braxton Hicks"],
        "precautions": ["Start folic acid 400mcg at least 1 month before conception","Avoid alcohol, smoking, raw fish, unpasteurised dairy","Attend all scheduled antenatal checkups","Sleep on left side – improves fetal blood flow","Avoid heavy lifting after 20 weeks","Monitor fetal movements daily from 28 weeks"],
        "diet_plan": ["🥛 Dairy: 3 servings/day for calcium (baby's bones)","🥦 Dark leafy greens: folate, iron, calcium","🍊 Citrus fruits: Vitamin C enhances iron absorption","🥚 Eggs: choline for fetal brain development","🐟 Safe fish: salmon, sardines (DHA for brain)","💧 Water: at least 2.5 litres/day","❌ Avoid: raw eggs, liver (excess Vit A), papaya, pineapple"],
        "yoga_poses": ["🤰 Cat-Cow (gentle, all trimesters)","🤰 Warrior II (1st & 2nd trimester only)","🤰 Prenatal Child's Pose (modified)","🤰 Seated Forward Bend (gentle)","🤰 Kegel exercises (pelvic floor strength)","⚠ Always practise with certified prenatal yoga instructor"],
        "medicines": ["Folic Acid 400–800 mcg (prevent neural tube defects)","Iron 27mg + Vitamin C (prevent anaemia)","Calcium 1000mg + Vitamin D3","DHA / Omega-3 (fetal brain development)","⚕ Prescribed: Iron infusion, anti-emetics if needed"],
        "doctors": ["Obstetrician / Gynaecologist","Nutritionist","Physiotherapist (pelvic floor)","Paediatrician (newborn care)"]
    },
    "irregular periods": {
        "age_groups": "Teens, reproductive age women, perimenopause",
        "causes": ["PCOS / PCOD","Thyroid disorders (hypo or hyper)","Stress and anxiety","Sudden weight gain or loss","Over-exercise","Perimenopause","Uterine fibroids/polyps"],
        "symptoms": ["Cycles shorter than 21 days or longer than 35 days","Missed periods (oligomenorrhoea or amenorrhoea)","Very heavy or very light flow","Severe cramping","Spotting between periods","PMS symptoms intensified"],
        "precautions": ["Track period with an app (days, flow, symptoms)","Manage stress – cortisol directly suppresses reproductive hormones","Maintain healthy BMI (extremes both disrupt periods)","Get tested: thyroid, prolactin, testosterone, FSH, LH","Avoid extreme diets or over-exercising"],
        "diet_plan": ["🌿 Chasteberry (Vitex) tea – regulates LH naturally","🥬 Iron-rich foods during heavy flow: spinach, beetroot","🌾 Seed cycling protocol throughout cycle","🍫 Dark chocolate 70%+ – magnesium reduces cramps","🌿 Ginger tea – reduces prostaglandins (cramping)","❌ Avoid: excess soy, processed foods, caffeine"],
        "yoga_poses": ["🌸 Reclining Bound Angle Pose","🌸 Seated Forward Fold – stimulates ovaries","🌸 Cobra Pose – hormonal regulation","🌸 Head-to-Knee Pose (Janu Sirsasana)","🌸 Shoulder Stand (stimulates thyroid & ovaries)"],
        "medicines": ["Vitamin D3 + K2","Magnesium 300–400mg","Chasteberry / Vitex extract","Omega-3 fatty acids","⚕ Prescribed: Hormonal therapy, OCP, Progesterone"],
        "doctors": ["Gynaecologist","Endocrinologist","Nutritionist"]
    },
    "hair problems": {
        "age_groups": "Teens to post-menopausal women",
        "causes": ["Nutritional deficiencies (iron, biotin, zinc, protein)","PCOS / hormonal imbalance","Thyroid disorders","Post-pregnancy hormonal drop (telogen effluvium)","Chemical treatments / heat styling","Alopecia areata (autoimmune)"],
        "symptoms": ["Excessive hair on pillow / in shower drain","Thinning at temples or crown","Receding hairline","Brittle, dry, or frizzy hair","Scalp itching or dandruff","Loss of more than 100 hairs/day"],
        "precautions": ["Blood test: iron, ferritin, B12, D3, thyroid, testosterone","Use wide-tooth comb on wet hair; never brush wet","Avoid tight hairstyles (ponytails, braids) daily","Oil scalp 2x/week (coconut, castor, onion oil)","Switch to sulphate-free, gentle shampoo"],
        "diet_plan": ["🥚 Eggs: biotin + protein – #1 hair food","🌿 Spinach: iron, folate, Vitamins A & C","🥜 Almonds & walnuts: biotin, Vitamin E, zinc","🐟 Fatty fish: omega-3 for scalp health","🫘 Lentils: protein + biotin + iron combo","🌸 Amla (Indian gooseberry): Vitamin C richest food","❌ Avoid: crash diets, excess Vitamin A supplements"],
        "yoga_poses": ["💆 Downward Dog – increases scalp blood flow","💆 Headstand (Sirsasana) – if experienced practitioner","💆 Rabbit Pose (Sasangasana) – stimulates scalp","💆 Camel Pose – balances thyroid","💆 Kapalbhati pranayama 10 min daily"],
        "medicines": ["Biotin 5000–10000 mcg/day","Iron (Ferrous Gluconate) if ferritin < 70","Vitamin D3 2000 IU","Zinc 25–50mg","Saw Palmetto (if DHT-related loss)","⚕ Topical: Minoxidil 2–5% (prescribed)"],
        "doctors": ["Dermatologist / Trichologist","Endocrinologist","Gynaecologist (if hormonal)"]
    },
    "skin problems / acne": {
        "age_groups": "Teens, young adults, hormonal acne in 20s–30s",
        "causes": ["Excess sebum production","Bacterial overgrowth (C. acnes)","Hormonal changes (PCOS, puberty, menstrual cycle)","Dairy and high-GI foods","Stress (cortisol spikes)","Wrong skincare products","Pollution / sun damage"],
        "symptoms": ["Blackheads, whiteheads, cysts","Red inflamed pimples","Post-acne dark spots (PIH)","Oily T-zone","Dry patches + breakouts (combination skin)","Skin texture and enlarged pores"],
        "precautions": ["Never pick or squeeze pimples – worsens scarring","Change pillowcase every 3–4 days","Apply SPF 30+ every morning (non-comedogenic)","Remove makeup fully before sleeping","Keep hair products away from skin"],
        "diet_plan": ["💧 Water 3L/day – flushes toxins","🥦 Zinc-rich foods: pumpkin seeds, chickpeas","🫐 Antioxidant berries – fight inflammation","🌿 Green tea (drink + apply as toner)","🐟 Omega-3: salmon, flaxseeds – reduce inflammation","❌ Avoid: dairy, white sugar, whey protein, fried food","❌ Limit: high-GI foods (white rice, bread, sweets)"],
        "yoga_poses": ["✨ Pranayama (breathing) – oxygenates blood","✨ Forward bends – improve circulation to face","✨ Fish Pose – stimulates thyroid, improves skin","✨ Twisting poses – liver detox","✨ Shoulder Stand – hormonal balance"],
        "medicines": ["Niacinamide 4–10% serum (topical)","Salicylic Acid 2% cleanser","Zinc supplement 30–50mg","Vitamin C serum + SPF","⚕ Prescribed: Clindamycin, Retinoids, Isotretinoin, OCP"],
        "doctors": ["Dermatologist","Gynaecologist (if hormonal acne)","Nutritionist"]
    },
    "thyroid problems": {
        "age_groups": "Women aged 20–60 (5–8x more common in women than men)",
        "causes": ["Autoimmune (Hashimoto's – hypothyroid; Graves' – hyperthyroid)","Iodine deficiency or excess","Radiation therapy history","Genetic predisposition","Pregnancy (postpartum thyroiditis)","Certain medications (lithium, amiodarone)"],
        "symptoms": ["Hypothyroid: fatigue, weight gain, cold intolerance, constipation, dry skin","Hyperthyroid: weight loss, heat intolerance, rapid heartbeat, anxiety","Both: hair loss, irregular periods, mood changes","Goitre (enlarged thyroid gland)"],
        "precautions": ["Test TSH, T3, T4, TPO antibodies annually","Take thyroid medication on empty stomach, 30–60 min before food","Don't take with calcium, iron or antacids – blocks absorption","Manage stress – cortisol suppresses thyroid function","Avoid excessive raw goitrogenic foods (broccoli, cabbage)"],
        "diet_plan": ["🐟 Seaweed / seafood: iodine for hypothyroid","🥩 Lean meats + eggs: selenium (Brazil nuts too)","🌾 Gluten-free if Hashimoto's confirmed","🫘 Avoid unfermented soy – blocks iodine uptake","🌿 Ashwagandha: adaptogen that supports thyroid","❌ Raw goitrogens in excess: cabbage, cauliflower (cook them)"],
        "yoga_poses": ["🦋 Shoulder Stand (Sarvangasana) – directly stimulates thyroid","🦋 Fish Pose (Matsyasana) – stretches thyroid area","🦋 Plough Pose (Halasana)","🦋 Camel Pose – opens throat chakra","🦋 Lion's Breath – stimulates thyroid gland"],
        "medicines": ["Levothyroxine (T4 replacement) – hypothyroid","Selenium 200mcg (supports conversion T4→T3)","Vitamin D3 + B12 (commonly deficient)","Zinc 25mg","⚕ Hyperthyroid: Methimazole, Radioiodine, Surgery"],
        "doctors": ["Endocrinologist","Gynaecologist (if fertility affected)","Nutritionist"]
    },
    "diabetes / blood sugar": {
        "age_groups": "Gestational: pregnant women; Type 2: 30+; Type 1: any age",
        "causes": ["Insulin resistance (Type 2)","Autoimmune destruction of beta cells (Type 1)","Pregnancy hormones (gestational)","PCOS-related insulin resistance","Obesity, sedentary lifestyle","Genetic predisposition"],
        "symptoms": ["Frequent thirst and urination","Unexplained fatigue","Blurry vision","Slow-healing wounds","Tingling in feet/hands","Frequent infections (UTI, yeast)"],
        "precautions": ["Monitor fasting blood sugar and HbA1c every 3 months","Never skip meals – eat at regular intervals","Check feet daily for cuts, sores, or numbness","Exercise for 30 min daily – lowers blood sugar naturally","Test before and after meals to understand food responses"],
        "diet_plan": ["🌾 Low-GI grains: oats, barley, quinoa, millets","🥦 Non-starchy vegetables at every meal","🫘 Legumes: chickpeas, lentils, rajma","🍎 Low-GI fruits: berries, guava, apple, pear","🥩 Protein with every meal: slows glucose absorption","🌿 Cinnamon 1 tsp/day: lowers blood sugar naturally","❌ Avoid: white rice, sugary drinks, sweets, potatoes, fruit juice"],
        "yoga_poses": ["🔄 Bow Pose (Dhanurasana) – stimulates pancreas","🔄 Seated Forward Fold – calms nervous system","🔄 Supine Spinal Twist – massages abdominal organs","🔄 Legs Up the Wall – improves circulation","🔄 Kapalbhati + Anulom Vilom pranayama"],
        "medicines": ["Chromium Picolinate 400mcg (improves insulin sensitivity)","Berberine 500mg (comparable to Metformin in studies)","Magnesium Glycinate 300mg","Alpha-Lipoic Acid 600mg","⚕ Prescribed: Metformin, Insulin, GLP-1 agonists"],
        "doctors": ["Endocrinologist / Diabetologist","Nutritionist","Cardiologist","Ophthalmologist"]
    },
    "anaemia / iron deficiency": {
        "age_groups": "Most common in women of reproductive age, teen girls, pregnant women",
        "causes": ["Heavy menstrual bleeding","Poor dietary iron intake","Pregnancy (increased iron demand)","Malabsorption (celiac, IBS)","Vegetarian/vegan diet without planning","Chronic inflammation"],
        "symptoms": ["Extreme fatigue and weakness","Pale skin and conjunctiva","Shortness of breath on exertion","Cold hands and feet","Brittle nails / koilonychia (spoon nails)","Restless legs at night","Difficulty concentrating (brain fog)"],
        "precautions": ["Test: CBC, Serum Ferritin, Serum Iron, TIBC","Take iron supplement on empty stomach with Vitamin C","Never take iron with calcium, tea, or coffee","Cook in cast iron pan – transfers dietary iron to food","Address root cause (heavy periods → see gynaecologist)"],
        "diet_plan": ["🥬 Spinach + lemon juice – iron + absorption enhancer","🫘 Rajma, masoor dal, chickpeas – plant iron sources","🥩 Red meat 2–3x/week (if non-vegetarian) – heme iron","🌿 Beetroot + carrot juice daily","🍊 Vitamin C with every iron-rich meal","🌰 Pumpkin seeds, sesame seeds – iron-rich snacks","❌ Avoid: tea/coffee with meals (tannins block iron)"],
        "yoga_poses": ["❤️ Legs Up the Wall – improves blood circulation","❤️ Supported Bridge Pose – energising","❤️ Seated Forward Fold – calming, restorative","❤️ Gentle Cobra – stimulates digestive organs","❤️ Deep belly breathing – oxygenation"],
        "medicines": ["Ferrous Gluconate or Ferrous Bisglycinate (gentler on stomach)","Vitamin C 500mg with each iron dose","B12 (methylcobalamin 1000mcg) if also B12 deficient","Folate / Folic Acid 400–800mcg","⚕ Severe: IV Iron infusion or Blood transfusion"],
        "doctors": ["Gynaecologist","Haematologist","Gastroenterologist (if malabsorption)","Nutritionist"]
    },
    "depression / mood disorders": {
        "age_groups": "Any age; peaks in postpartum, perimenopause, adolescence",
        "causes": ["Serotonin, dopamine, norepinephrine imbalance","Hormonal fluctuations (postpartum, PMDD, menopause)","Thyroid dysfunction","Chronic stress or trauma","Vitamin D, B12, omega-3 deficiency","Social isolation"],
        "symptoms": ["Persistent sadness or emptiness","Loss of interest in activities","Fatigue and sleep disturbances (too much or too little)","Appetite changes – weight gain or loss","Difficulty concentrating","Feelings of worthlessness","Thoughts of self-harm (seek emergency help immediately)"],
        "precautions": ["Seek professional help – depression is a medical condition","Don't isolate – reach out to 1 trusted person daily","Establish routine: wake, eat, sleep at consistent times","Exercise is as effective as antidepressants for mild depression","Limit alcohol – it worsens depressive symptoms"],
        "diet_plan": ["🐟 Fatty fish 3x/week: omega-3 (EPA/DHA) – brain health","🍫 Dark chocolate: phenylethylamine boosts mood","🫐 Berries: polyphenols protect brain cells","🌿 Turmeric: curcumin as effective as antidepressants in studies","🥬 Leafy greens: folate for serotonin synthesis","🍌 Banana + yoghurt: tryptophan → serotonin","❌ Avoid: alcohol, ultra-processed foods, excess sugar"],
        "yoga_poses": ["🌞 Sun Salutation – energises, boosts serotonin","🌞 Backbends (Camel, Fish, Cobra) – heart openers","🌞 Inversions: Legs Up the Wall – shifts perspective","🌞 Yoga Nidra (body scan meditation) – 20 min","🌞 Dance / movement therapy (Nritya Yoga)"],
        "medicines": ["Omega-3 (EPA 1000mg+) – reduces depression symptoms","Vitamin D3 2000–4000 IU (test levels first)","St. John's Wort (mild–moderate; drug interactions – consult doctor)","Saffron extract (clinical studies support mood benefits)","⚕ Prescribed: SSRIs, SNRIs, Therapy (CBT/DBT)"],
        "doctors": ["Psychiatrist","Psychologist / Therapist","Endocrinologist (if hormonal)","Neurologist"]
    },
    "menopause": {
        "age_groups": "Women aged 45–55 (perimenopause may start at 40)",
        "causes": ["Natural decline in oestrogen and progesterone production","Surgical menopause (after oophorectomy)"],
        "symptoms": ["Hot flashes and night sweats","Irregular then absent periods","Vaginal dryness","Mood swings, anxiety, depression","Insomnia","Weight gain (especially belly)","Reduced libido","Brain fog","Joint pain"],
        "precautions": ["Keep bedroom cool (18–19°C) for night sweats","Wear breathable natural fabrics","Avoid triggers: spicy food, caffeine, alcohol","Do weight-bearing exercise to protect bone density","Maintain vaginal health with moisturisers and lubricants"],
        "diet_plan": ["🫘 Soy isoflavones: tofu, edamame, miso (phytoestrogens)","🥛 Calcium: dairy, fortified plant milk, almonds","🐟 Fatty fish: omega-3 for heart & mood","🌾 Flaxseeds: lignans balance hormones","🍎 Fibre-rich foods prevent weight gain","❌ Limit: alcohol, caffeine, sugary foods (worsen hot flashes)"],
        "yoga_poses": ["🌙 Yin Yoga: long-held, cooling poses","🌙 Moon Salutation (Chandra Namaskar)","🌙 Seated Wide-Angle Forward Fold","🌙 Restorative Yoga: bolster-supported poses","🌙 Shavasana with guided relaxation"],
        "medicines": ["Soy isoflavones 40–80mg (non-hormonal option)","Black Cohosh extract","Vitamin D3 2000 IU + Calcium 1200mg","Magnesium Glycinate (sleep + mood)","⚕ HRT: Oestrogen/Progesterone therapy (discuss risks with doctor)"],
        "doctors": ["Gynaecologist / Menopause Specialist","Endocrinologist","Cardiologist","Bone Density Specialist"]
    },
    "migraine / headache": {
        "age_groups": "Women 3x more likely than men; peaks ages 20–45",
        "causes": ["Hormonal changes (oestrogen fluctuations around periods)","Stress and sleep disruption","Dietary triggers: tyramine, MSG, alcohol, caffeine withdrawal","Sensory triggers: bright light, strong smells","Dehydration","Weather/pressure changes"],
        "symptoms": ["Throbbing pain (usually one side)","Nausea and vomiting","Light and sound sensitivity","Aura: visual disturbances, tingling","Neck stiffness before attack","Cognitive fog (migraine hangover)"],
        "precautions": ["Keep a migraine diary (identify patterns and triggers)","Sleep consistent hours – even weekends","Stay hydrated: 2–3 litres water daily","Avoid skipping meals – hypoglycaemia triggers migraine","Wear sunglasses in bright light / use blue-light filters"],
        "diet_plan": ["💧 Water first on waking – dehydration is #1 trigger","🥜 Magnesium-rich: almonds, pumpkin seeds, dark leafy greens","🐟 Omega-3: reduce frequency of migraines","🍒 Cherries / tart cherry juice – anti-inflammatory","☕ Small amount of caffeine can help during attack","❌ Avoid: red wine, aged cheese, processed meats, MSG, aspartame"],
        "yoga_poses": ["🧊 Child's Pose with forehead on block","🧊 Legs Up the Wall (during attack – dark room)","🧊 Supported Reclining Bound Angle","🧊 Neck and shoulder stretches (prevent tension migraines)","🧊 Alternate Nostril Breathing"],
        "medicines": ["Magnesium Glycinate 400mg daily (preventive)","Riboflavin (B2) 400mg daily (preventive)","CoQ10 300mg daily (preventive)","Melatonin 3mg at bedtime","⚕ Prescribed: Triptans (acute), Topiramate/Propranolol (preventive)"],
        "doctors": ["Neurologist","Gynaecologist (if menstrual migraine)","Ophthalmologist","Pain Management Specialist"]
    },
    "back pain / posture": {
        "age_groups": "All ages; most common in working women, pregnant women, post-delivery",
        "causes": ["Prolonged sitting with poor posture","Weak core muscles","Heavy lifting (including newborn care)","Pregnancy-related postural shift","Disc herniation or sciatica","Osteoporosis in older women"],
        "symptoms": ["Dull aching lower or upper back pain","Pain after sitting long hours","Radiating pain to legs (sciatica)","Morning stiffness","Pain worsening with bending or lifting","Poor posture (rounded shoulders)"],
        "precautions": ["Set screen at eye level; chair should support lumbar curve","Stand and stretch for 2 min every 30 min","Lift with bent knees, not bent back","Sleep on firm mattress; side-lie with pillow between knees","Strengthen core before starting exercise programs"],
        "diet_plan": ["🐟 Omega-3: salmon, walnuts – reduce spinal inflammation","🥛 Calcium: dairy, broccoli, almonds – bone strength","🌿 Turmeric + ginger: natural anti-inflammatories","🍒 Cherries: collagen and anti-inflammatory","💧 Adequate hydration – maintains disc fluid balance","❌ Avoid: processed foods, excess sugar (pro-inflammatory)"],
        "yoga_poses": ["🌿 Cat-Cow Stretch – spinal mobility","🌿 Child's Pose – lower back release","🌿 Supine Twist – relieves sciatica","🌿 Pigeon Pose – hip flexor stretch","🌿 Bridge Pose – strengthens lower back & glutes","🌿 Downward Dog – decompresses spine"],
        "medicines": ["Magnesium Glycinate (muscle relaxant)","Turmeric Curcumin 500–1000mg","Vitamin D3 + K2 (bone health)","Topical: Diclofenac gel","⚕ Prescribed: Muscle relaxants, NSAIDs, Physiotherapy, Injections"],
        "doctors": ["Physiotherapist","Orthopaedic Surgeon","Spine Specialist","Chiropractor","Neurologist (if sciatica)"]
    },
    "childhood / teen health (girls)": {
        "age_groups": "Girls aged 8–18 (puberty to late teens)",
        "causes": ["Nutritional deficiencies during growth spurts","Body image issues","Academic stress","Social pressure"],
        "symptoms": ["Irregular periods in first 2 years (normal)","Acne during puberty","Growth spurts and bone pain","Mood swings (hormonal)","Low energy / fatigue (often iron deficiency)","Body image concerns"],
        "precautions": ["Ensure calcium + Vitamin D for bone building (peak bone mass by 25)","Educate on menstrual hygiene and cycle tracking","Screen time limits – affects sleep and mental health","Encourage sport and physical activity","Open conversations about body changes and mental wellbeing"],
        "diet_plan": ["🥛 Dairy or fortified plant milk: calcium for growing bones","🥚 Eggs + dal: protein for growth and hormones","🥬 Iron-rich foods: spinach, rajma (onset of menstruation increases needs)","🍊 Citrus daily: Vitamin C + immunity","🌰 Healthy snacks: nuts, seeds, fruits instead of chips","❌ Limit: sugary drinks, energy drinks, junk food"],
        "yoga_poses": ["🌈 Tree Pose (balance & focus)","🌈 Warrior poses (confidence & strength)","🌈 Forward folds (calm exam anxiety)","🌈 Happy Baby Pose (relaxation)","🌈 Dance yoga / fun movement classes"],
        "medicines": ["Multivitamin with iron for teen girls","Calcium 1300mg + Vitamin D3 600 IU","Omega-3 (DHA/EPA) for brain development","B-Complex for stress and energy","⚕ Consult paediatrician for personalised guidance"],
        "doctors": ["Paediatrician / Adolescent Medicine","Gynaecologist (first visit by 13–15)","Nutritionist","School Counsellor"]
    },
}

PROBLEM_MAP = {
    "Stress / Anxiety": "stress / anxiety",
    "Weight Gain / Obesity": "weight gain / obesity",
    "Weight Loss / Underweight": "weight loss / underweight",
    "PCOD / PCOS": "pcod / pcos",
    "Pregnancy / Prenatal Care": "pregnancy / prenatal care",
    "Irregular Periods": "irregular periods",
    "Hair Problems": "hair problems",
    "Skin Problems / Acne": "skin problems / acne",
    "Thyroid Problems": "thyroid problems",
    "Diabetes / Blood Sugar": "diabetes / blood sugar",
    "Anaemia / Iron Deficiency": "anaemia / iron deficiency",
    "Depression / Mood Disorders": "depression / mood disorders",
    "Menopause": "menopause",
    "Migraine / Headache": "migraine / headache",
    "Back Pain / Posture": "back pain / posture",
    "Childhood / Teen Health (Girls)": "childhood / teen health (girls)",
}

# ══════════════════════════════════════════════
#                  HELPERS
# ══════════════════════════════════════════════
def hash_pw(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def bmi_calc(weight, height_cm):
    if height_cm <= 0:
        return 0, "N/A"
    h = height_cm / 100
    bmi = round(weight / (h * h), 1)
    if bmi < 18.5: cat = "Underweight"
    elif bmi < 25: cat = "Normal"
    elif bmi < 30: cat = "Overweight"
    else: cat = "Obese"
    return bmi, cat

def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_email" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

# ══════════════════════════════════════════════
#                  HTML TEMPLATE
# ══════════════════════════════════════════════
BASE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>✦ AI Women's Health Analyser</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #0d0818;
  --bg2: #130f22;
  --card: #1a1030;
  --card2: #221540;
  --accent: #c084fc;
  --accent2: #f0abfc;
  --accent3: #818cf8;
  --text: #f3e8ff;
  --muted: #9d7ec9;
  --success: #4ade80;
  --warning: #facc15;
  --danger: #f87171;
  --info: #60a5fa;
  --entry: #1e1238;
  --btn: #7c3aed;
  --btn-hov: #9333ea;
  --border: #2d1a52;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: var(--bg); color: var(--text); font-family: 'DM Sans', sans-serif; min-height: 100vh; }
a { color: var(--accent); text-decoration: none; }
a:hover { color: var(--accent2); }

/* NAV */
nav {
  background: var(--card2);
  border-bottom: 1px solid var(--border);
  padding: 0 1.5rem;
  display: flex;
  align-items: center;
  gap: 0.25rem;
  height: 52px;
  position: sticky;
  top: 0;
  z-index: 100;
  backdrop-filter: blur(8px);
}
nav::before {
  content: '';
  position: absolute;
  left: 0; top: 0; bottom: 0;
  width: 3px;
  background: linear-gradient(180deg, var(--accent), var(--accent3));
}
.nav-brand { font-family: 'DM Serif Display', serif; color: var(--accent); font-size: 1.1rem; margin-right: 0.75rem; padding-left: 0.5rem; white-space: nowrap; }
.nav-links { display: flex; gap: 0.1rem; flex: 1; flex-wrap: wrap; }
.nav-links a {
  color: var(--muted); font-size: 0.82rem; padding: 0.4rem 0.7rem;
  border-radius: 6px; transition: all 0.2s; font-weight: 500; white-space: nowrap;
}
.nav-links a:hover, .nav-links a.active { color: var(--accent); background: rgba(192,132,252,0.1); }
.nav-logout { color: var(--danger) !important; margin-left: auto; }

/* CARDS */
.card { background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 1.5rem; margin-bottom: 1rem; }
.card-header { border-bottom: 1px solid var(--border); padding-bottom: 0.75rem; margin-bottom: 1rem; }
.section-title { color: var(--accent2); font-weight: 600; font-size: 0.95rem; letter-spacing: 0.03em; display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.75rem; padding-bottom: 0.5rem; border-bottom: 1px solid var(--border); }

/* FORMS */
.form-group { margin-bottom: 1rem; }
.form-group label { display: block; color: var(--accent2); font-size: 0.85rem; font-weight: 600; margin-bottom: 0.4rem; }
input, select, textarea {
  width: 100%; background: var(--entry); border: 1px solid var(--border); color: var(--text);
  padding: 0.6rem 0.9rem; border-radius: 8px; font-family: 'DM Sans', sans-serif; font-size: 0.9rem;
  outline: none; transition: border-color 0.2s;
}
input:focus, select:focus, textarea:focus { border-color: var(--accent); }
select option { background: var(--card); }
textarea { resize: vertical; min-height: 100px; }

/* BUTTONS */
.btn {
  background: var(--btn); color: var(--text); border: none; padding: 0.65rem 1.4rem;
  border-radius: 8px; font-family: 'DM Sans', sans-serif; font-weight: 600; font-size: 0.9rem;
  cursor: pointer; transition: all 0.2s; display: inline-flex; align-items: center; gap: 0.4rem;
}
.btn:hover { background: var(--btn-hov); transform: translateY(-1px); }
.btn-success { background: #059669; } .btn-success:hover { background: #10b981; }
.btn-danger { background: #dc2626; } .btn-danger:hover { background: #ef4444; }
.btn-outline { background: transparent; border: 1px solid var(--border); }
.btn-outline:hover { border-color: var(--accent); background: rgba(192,132,252,0.1); }
.btn-sm { padding: 0.4rem 0.9rem; font-size: 0.82rem; }

/* PAGE LAYOUT */
.page-wrap { max-width: 980px; margin: 0 auto; padding: 1.5rem 1rem 3rem; }
.page-header { text-align: center; padding: 1.5rem 0 1rem; }
.page-header h1 { font-family: 'DM Serif Display', serif; font-size: 2rem; color: var(--text); }
.page-header p { color: var(--muted); margin-top: 0.3rem; font-size: 0.9rem; }

/* STAT CARDS */
.stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1.5rem; }
.stat-card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 1.2rem; text-align: center; }
.stat-val { font-size: 2rem; font-weight: 700; color: var(--accent); font-family: 'DM Serif Display', serif; }
.stat-lbl { font-size: 0.78rem; color: var(--muted); margin-top: 0.2rem; }

/* ALERTS */
.alert { padding: 0.75rem 1rem; border-radius: 8px; font-size: 0.88rem; margin-bottom: 1rem; }
.alert-success { background: rgba(74,222,128,0.1); border: 1px solid rgba(74,222,128,0.3); color: var(--success); }
.alert-danger { background: rgba(248,113,113,0.1); border: 1px solid rgba(248,113,113,0.3); color: var(--danger); }
.alert-warning { background: rgba(250,204,21,0.1); border: 1px solid rgba(250,204,21,0.3); color: var(--warning); }
.alert-info { background: rgba(96,165,250,0.1); border: 1px solid rgba(96,165,250,0.3); color: var(--info); }

/* TABLES */
table { width: 100%; border-collapse: collapse; font-size: 0.87rem; }
th { background: var(--card2); color: var(--accent2); font-weight: 600; padding: 0.7rem 0.9rem; text-align: left; }
td { padding: 0.65rem 0.9rem; border-bottom: 1px solid var(--border); color: var(--text); }
tr:hover td { background: rgba(192,132,252,0.05); }

/* BULLET LIST */
.bullet-list { list-style: none; }
.bullet-list li { padding: 0.3rem 0; padding-left: 1.4rem; position: relative; font-size: 0.9rem; line-height: 1.5; }
.bullet-list li::before { content: '▸'; position: absolute; left: 0; color: var(--accent); }

/* GRID */
.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 1.2rem; }
.three-col { display: grid; grid-template-columns: repeat(3,1fr); gap: 1rem; }
@media (max-width: 700px) { .two-col, .three-col, .stats-grid { grid-template-columns: 1fr; } }

/* BADGE */
.badge { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 999px; font-size: 0.75rem; font-weight: 600; }
.badge-success { background: rgba(74,222,128,0.15); color: var(--success); }
.badge-warning { background: rgba(250,204,21,0.15); color: var(--warning); }
.badge-danger  { background: rgba(248,113,113,0.15); color: var(--danger); }
.badge-info    { background: rgba(96,165,250,0.15); color: var(--info); }
.badge-accent  { background: rgba(192,132,252,0.15); color: var(--accent); }

/* METRIC BARS */
.metric-bar-bg { background: var(--border); border-radius: 999px; height: 10px; overflow: hidden; margin: 0.3rem 0; }
.metric-bar-fill { height: 100%; border-radius: 999px; transition: width 1s ease; }

/* MONO */
.mono { font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; }

/* STEP INDICATOR */
.steps { display: flex; gap: 0; margin-bottom: 1.5rem; }
.step { flex: 1; text-align: center; padding: 0.5rem 0; font-size: 0.78rem; color: var(--muted); border-bottom: 2px solid var(--border); position: relative; }
.step.active { color: var(--accent); border-color: var(--accent); font-weight: 600; }
.step.done { color: var(--success); border-color: var(--success); }

/* LOGIN SPECIAL */
.login-wrap { min-height: 100vh; display: flex; align-items: center; justify-content: center; padding: 2rem; }
.login-box { width: 100%; max-width: 420px; }
.logo-big { font-family: 'DM Serif Display', serif; font-size: 2.2rem; color: var(--accent); text-align: center; margin-bottom: 0.25rem; }
.logo-sub { text-align: center; color: var(--muted); font-size: 0.88rem; margin-bottom: 1.5rem; }
</style>
</head>
<body>
{% if show_nav %}
<nav>
  <span class="nav-brand">✦ WellnessAI</span>
  <div class="nav-links">
    <a href="{{ url_for('dashboard') }}" class="{{ 'active' if active_page=='dashboard' }}">🏠 Home</a>
    <a href="{{ url_for('details') }}" class="{{ 'active' if active_page=='details' }}">📋 Details</a>
    <a href="{{ url_for('analysis') }}" class="{{ 'active' if active_page=='analysis' }}">🔬 Analysis</a>
    <a href="{{ url_for('bmi') }}" class="{{ 'active' if active_page=='bmi' }}">⚖ BMI</a>
    <a href="{{ url_for('cycle') }}" class="{{ 'active' if active_page=='cycle' }}">🌸 Cycle</a>
    <a href="{{ url_for('appointments') }}" class="{{ 'active' if active_page=='appointments' }}">📅 Appointments</a>
    <a href="{{ url_for('history') }}" class="{{ 'active' if active_page=='history' }}">📂 History</a>
    <a href="{{ url_for('ml_metrics') }}" class="{{ 'active' if active_page=='ml_metrics' }}">📊 ML Metrics</a>
    <a href="{{ url_for('feedback') }}" class="{{ 'active' if active_page=='feedback' }}">⭐ Feedback</a>
    <a href="{{ url_for('logout') }}" class="nav-logout">🚪 Logout</a>
  </div>
</nav>
{% endif %}
{{ content }}
</body>
</html>"""

def render_page(content, active_page="", show_nav=True):
    return render_template_string(BASE_HTML, content=content, active_page=active_page, show_nav=show_nav)

# ══════════════════════════════════════════════
#                   ROUTES
# ══════════════════════════════════════════════

@app.route("/")
def index():
    return redirect(url_for("login"))

# ── LOGIN ──
@app.route("/login", methods=["GET", "POST"])
def login():
    error = ""
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        pw = request.form.get("password", "")
        db = get_db()
        row = db.execute("SELECT * FROM users WHERE email=? AND password=?", (email, hash_pw(pw))).fetchone()
        db.close()
        if row:
            session["user_email"] = row["email"]
            session["user_name"] = row["name"]
            session["user_height"] = row["height"]
            session["user_weight"] = row["weight"]
            session["user_age"] = row["age"]
            session["user_blood"] = row["blood_group"]
            return redirect(url_for("dashboard"))
        else:
            error = "Invalid email or password."

    quotes = ["Your health is your wealth.", "Self-care is not selfish.", "Strong women lift each other up."]
    quote = random.choice(quotes)
    html = f"""
<div class="login-wrap">
  <div class="login-box">
    <div class="logo-big">✦</div>
    <div class="logo-big" style="font-size:1.6rem">AI Women's Health</div>
    <div class="logo-sub">Your personalised wellness companion</div>
    {'<div class="alert alert-danger">' + error + '</div>' if error else ''}
    <div class="card">
      <form method="POST">
        <div class="form-group">
          <label>Email Address</label>
          <input type="email" name="email" required placeholder="you@email.com">
        </div>
        <div class="form-group">
          <label>Password</label>
          <input type="password" name="password" required placeholder="••••••••">
        </div>
        <button class="btn" style="width:100%;justify-content:center" type="submit">Login →</button>
        <hr style="border-color:var(--border);margin:1rem 0">
        <p style="text-align:center;color:var(--muted);font-size:0.85rem;margin-bottom:0.75rem">New here?</p>
        <a href="{url_for('register')}" class="btn btn-outline" style="width:100%;justify-content:center">Create Account</a>
      </form>
    </div>
    <p style="text-align:center;color:var(--muted);font-size:0.82rem;font-style:italic;margin-top:1rem">"{quote}"</p>
  </div>
</div>"""
    return render_page(html, show_nav=False)

# ── REGISTER ──
@app.route("/register", methods=["GET", "POST"])
def register():
    error = ""; success = ""
    if request.method == "POST":
        vals = {k: request.form.get(k, "").strip() for k in
                ["email","password","name","age","address","blood_group","height","weight"]}
        if any(not v for v in vals.values()):
            error = "Please fill all fields."
        elif len(vals["password"]) < 6:
            error = "Password must be at least 6 characters."
        else:
            try:
                h, w = float(vals["height"]), float(vals["weight"])
                db = get_db()
                db.execute("INSERT INTO users(email,password,name,age,address,blood_group,height,weight,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                    (vals["email"], hash_pw(vals["password"]), vals["name"], vals["age"],
                     vals["address"], vals["blood_group"], h, w,
                     datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
                db.commit(); db.close()
                success = f"Welcome, {vals['name']}! Please login."
            except sqlite3.IntegrityError:
                error = "Email already registered."
            except ValueError:
                error = "Height and Weight must be numbers."

    html = f"""
<div class="page-wrap" style="max-width:520px">
  <div class="page-header">
    <div class="logo-big" style="font-family:'DM Serif Display',serif;font-size:1.8rem;color:var(--accent)">Create Account</div>
    <p class="logo-sub" style="color:var(--muted)">Join the WellnessAI community</p>
  </div>
  {'<div class="alert alert-danger">' + error + '</div>' if error else ''}
  {'<div class="alert alert-success">' + success + ' <a href="' + url_for('login') + '">Login now →</a></div>' if success else ''}
  <div class="card">
    <form method="POST">
      <div class="two-col">
        <div class="form-group"><label>📧 Email</label><input type="email" name="email" required></div>
        <div class="form-group"><label>🔒 Password</label><input type="password" name="password" required></div>
        <div class="form-group"><label>👤 Full Name</label><input name="name" required></div>
        <div class="form-group"><label>🎂 Age</label><input name="age" required></div>
        <div class="form-group"><label>🩸 Blood Group</label>
          <select name="blood_group">
            <option>A+</option><option>A-</option><option>B+</option><option>B-</option>
            <option>O+</option><option>O-</option><option>AB+</option><option>AB-</option>
          </select>
        </div>
        <div class="form-group"><label>🏠 Address</label><input name="address" required></div>
        <div class="form-group"><label>📏 Height (cm)</label><input type="number" step="0.1" name="height" required></div>
        <div class="form-group"><label>⚖ Weight (kg)</label><input type="number" step="0.1" name="weight" required></div>
      </div>
      <button class="btn" style="width:100%;justify-content:center" type="submit">✅ Register</button>
      <div style="text-align:center;margin-top:0.75rem">
        <a href="{url_for('login')}" class="btn btn-outline btn-sm">← Back to Login</a>
      </div>
    </form>
  </div>
</div>"""
    return render_page(html, show_nav=False)

# ── LOGOUT ──
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ── DASHBOARD ──
@app.route("/dashboard")
@login_required
def dashboard():
    email = session["user_email"]
    name = session.get("user_name", "User")
    db = get_db()
    rec = db.execute("SELECT COUNT(*) FROM health_records WHERE user_email=?", (email,)).fetchone()[0]
    appt = db.execute("SELECT COUNT(*) FROM appointments WHERE user_email=?", (email,)).fetchone()[0]
    cyc = db.execute("SELECT COUNT(*) FROM cycle_tracker WHERE user_email=?", (email,)).fetchone()[0]
    db.close()
    h = float(session.get("user_height") or 0)
    w = float(session.get("user_weight") or 0)
    bmi, cat = bmi_calc(w, h) if h and w else (0, "N/A")
    today = datetime.datetime.now().strftime("%A, %d %B %Y")
    quotes = ["Your body is worthy of care and attention.", "Small steps lead to big health changes.", "Knowledge is the first medicine."]
    bmi_col = {"Underweight":"var(--info)","Normal":"var(--success)","Overweight":"var(--warning)","Obese":"var(--danger)"}.get(cat,"var(--accent)")

    html = f"""
<div class="page-wrap">
  <div style="background:var(--card2);border-radius:14px;padding:1.5rem 2rem;margin-bottom:1.5rem;border:1px solid var(--border)">
    <h1 style="font-family:'DM Serif Display',serif;font-size:1.9rem;color:var(--accent)">Welcome back, {name} 💜</h1>
    <p style="color:var(--muted);margin-top:0.25rem">{today}</p>
  </div>

  <div class="stats-grid">
    <div class="stat-card"><div class="stat-val" style="color:var(--accent)">{rec}</div><div class="stat-lbl">📋 Health Records</div></div>
    <div class="stat-card"><div class="stat-val" style="color:var(--accent2)">{appt}</div><div class="stat-lbl">📅 Appointments</div></div>
    <div class="stat-card"><div class="stat-val" style="color:var(--success)">{cyc}</div><div class="stat-lbl">🌸 Cycle Logs</div></div>
    <div class="stat-card"><div class="stat-val" style="color:{bmi_col}">{bmi}</div><div class="stat-lbl">⚖ BMI – {cat}</div></div>
  </div>

  <div class="card" style="margin-bottom:1.5rem">
    <div class="section-title">✦ Your 5-Step Health Journey</div>
    <div class="steps">
      <div class="step done">1<br><small>Account</small></div>
      <div class="step done">2<br><small>Details</small></div>
      <div class="step done">3<br><small>Analysis</small></div>
      <div class="step done">4<br><small>Solutions</small></div>
      <div class="step">5<br><small>Feedback</small></div>
    </div>
  </div>

  <div class="three-col">
    <a href="{url_for('analysis')}" class="card" style="display:block;text-align:center;transition:transform 0.2s" onmouseover="this.style.transform='translateY(-3px)'" onmouseout="this.style.transform=''">
      <div style="font-size:2rem">🔬</div><div style="font-weight:600;color:var(--accent);margin-top:0.5rem">Health Analysis</div>
      <div style="font-size:0.82rem;color:var(--muted);margin-top:0.25rem">AI-powered diagnosis</div>
    </a>
    <a href="{url_for('bmi')}" class="card" style="display:block;text-align:center;transition:transform 0.2s" onmouseover="this.style.transform='translateY(-3px)'" onmouseout="this.style.transform=''">
      <div style="font-size:2rem">⚖</div><div style="font-weight:600;color:var(--accent2);margin-top:0.5rem">BMI Calculator</div>
      <div style="font-size:0.82rem;color:var(--muted);margin-top:0.25rem">Track your weight</div>
    </a>
    <a href="{url_for('cycle')}" class="card" style="display:block;text-align:center;transition:transform 0.2s" onmouseover="this.style.transform='translateY(-3px)'" onmouseout="this.style.transform=''">
      <div style="font-size:2rem">🌸</div><div style="font-weight:600;color:var(--success);margin-top:0.5rem">Cycle Tracker</div>
      <div style="font-size:0.82rem;color:var(--muted);margin-top:0.25rem">Menstrual health</div>
    </a>
    <a href="{url_for('appointments')}" class="card" style="display:block;text-align:center;transition:transform 0.2s" onmouseover="this.style.transform='translateY(-3px)'" onmouseout="this.style.transform=''">
      <div style="font-size:2rem">📅</div><div style="font-weight:600;color:var(--warning);margin-top:0.5rem">Appointments</div>
      <div style="font-size:0.82rem;color:var(--muted);margin-top:0.25rem">Book doctors</div>
    </a>
    <a href="{url_for('ml_metrics')}" class="card" style="display:block;text-align:center;transition:transform 0.2s" onmouseover="this.style.transform='translateY(-3px)'" onmouseout="this.style.transform=''">
      <div style="font-size:2rem">📊</div><div style="font-weight:600;color:var(--info);margin-top:0.5rem">ML Metrics</div>
      <div style="font-size:0.82rem;color:var(--muted);margin-top:0.25rem">Model performance</div>
    </a>
    <a href="{url_for('feedback')}" class="card" style="display:block;text-align:center;transition:transform 0.2s" onmouseover="this.style.transform='translateY(-3px)'" onmouseout="this.style.transform=''">
      <div style="font-size:2rem">⭐</div><div style="font-weight:600;color:var(--accent);margin-top:0.5rem">Feedback</div>
      <div style="font-size:0.82rem;color:var(--muted);margin-top:0.25rem">Share your experience</div>
    </a>
  </div>

  <p style="text-align:center;color:var(--muted);font-style:italic;margin-top:1.5rem;font-size:0.87rem">"{random.choice(quotes)}"</p>
</div>"""
    return render_page(html, active_page="dashboard")

# ── DETAILS ──
@app.route("/details", methods=["GET","POST"])
@login_required
def details():
    if request.method == "POST":
        session["problem"] = request.form.get("problem","")
        session["age_group"] = request.form.get("age_group","")
        session["duration"] = request.form.get("duration","")
        session["severity"] = request.form.get("severity","3")
        session["notes"] = request.form.get("notes","")
        return redirect(url_for("analysis"))

    html = f"""
<div class="page-wrap" style="max-width:640px">
  <div class="page-header"><h1>📋 Health Details</h1><p>STEP 2 of 5 – Select your primary health concern</p></div>
  <div class="card">
    <form method="POST">
      <div class="form-group">
        <label>Your Age Group</label>
        <select name="age_group">
          <option>Child (under 12)</option><option>Teen (12–18)</option>
          <option selected>Adult (18–45)</option><option>Middle Age (45–60)</option><option>Senior (60+)</option>
        </select>
      </div>
      <div class="form-group">
        <label>Primary Health Concern *</label>
        <select name="problem" required>
          <option value="">-- Select a concern --</option>
          {"".join(f'<option value="{k}">{k}</option>' for k in PROBLEM_MAP.keys())}
        </select>
      </div>
      <div class="form-group">
        <label>How long have you had this concern?</label>
        <select name="duration">
          <option>Less than 1 month</option><option>1–3 months</option>
          <option>3–6 months</option><option>6–12 months</option><option>More than 1 year</option>
        </select>
      </div>
      <div class="form-group">
        <label>Severity (1 = Mild, 5 = Severe)</label>
        <div style="display:flex;gap:1rem;margin-top:0.25rem">
          {"".join(f'<label style="display:flex;align-items:center;gap:0.3rem;cursor:pointer"><input type="radio" name="severity" value="{i}" {"checked" if i==3 else ""}> {i}</label>' for i in range(1,6))}
        </div>
      </div>
      <div class="form-group">
        <label>Additional Symptoms / Notes</label>
        <textarea name="notes" placeholder="Describe any other symptoms..."></textarea>
      </div>
      <div style="display:flex;gap:0.75rem;flex-wrap:wrap">
        <button class="btn" type="submit">Continue to Analysis →</button>
        <a href="{url_for('dashboard')}" class="btn btn-outline">← Dashboard</a>
      </div>
    </form>
  </div>
</div>"""
    return render_page(html, active_page="details")

# ── ANALYSIS ──
@app.route("/analysis")
@login_required
def analysis():
    problem = session.get("problem","")
    email = session["user_email"]
    name = session.get("user_name","N/A")
    age_group = session.get("age_group","N/A")
    blood = session.get("user_blood","N/A")
    severity = session.get("severity","N/A")
    duration = session.get("duration","N/A")
    notes_txt = session.get("notes","")
    h = float(session.get("user_height") or 0)
    w = float(session.get("user_weight") or 0)
    bmi, cat = bmi_calc(w, h) if h and w else (0, "N/A")
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    if not problem:
        html = f"""
<div class="page-wrap" style="max-width:600px">
  <div class="page-header"><h1>🔬 AI Health Analysis</h1></div>
  <div class="alert alert-warning">⚠ Please go to the <a href="{url_for('details')}">Details page</a> and select a health concern first.</div>
</div>"""
        return render_page(html, active_page="analysis")

    detail_key = PROBLEM_MAP.get(problem, "")
    detail = HEALTH_DETAILS.get(detail_key, {})
    bmi_col = {"Underweight":"var(--info)","Normal":"var(--success)","Overweight":"var(--warning)","Obese":"var(--danger)"}.get(cat,"var(--accent)")

    sections_html = ""
    section_defs = [
        ("⚠ Common Causes","causes","var(--text)"),
        ("🩺 Symptoms & Signs","symptoms","var(--accent2)"),
        ("🛡 Precautions & Lifestyle","precautions","var(--success)"),
        ("🥗 Recommended Diet Plan","diet_plan","var(--text)"),
        ("🧘 Yoga & Exercise","yoga_poses","var(--accent2)"),
        ("💊 Supplements & Medicines","medicines","var(--warning)"),
        ("🏥 Recommended Specialists","doctors","var(--info)"),
    ]
    for title, key, col in section_defs:
        items = detail.get(key,[])
        if items:
            lis = "".join(f'<li style="color:{col}">{i}</li>' for i in items)
            warning = '<p style="color:var(--danger);font-size:0.82rem;font-style:italic;margin-top:0.5rem">⚕ Always consult a doctor before starting any medication.</p>' if key=="medicines" else ""
            sections_html += f"""
<div class="card">
  <div class="section-title">{title}</div>
  <ul class="bullet-list">{lis}</ul>
  {warning}
</div>"""

    # Save to DB
    try:
        db = get_db()
        db.execute("INSERT INTO health_records(user_email,problem,solution,bmi,bmi_category,notes,recorded_at) VALUES(?,?,?,?,?,?,?)",
            (email, problem, detail.get("medicines",[""])[0] or "See report", bmi, cat, notes_txt, now))
        db.commit(); db.close()
    except: pass

    html = f"""
<div class="page-wrap">
  <div class="page-header"><h1>🔬 AI Health Analysis</h1><p>STEP 3–4 of 5 – AI-Generated Solutions</p></div>
  <div style="display:flex;gap:0.75rem;margin-bottom:1.5rem;flex-wrap:wrap">
    <a href="{url_for('details')}" class="btn btn-outline btn-sm">← Select Problem</a>
    <a href="{url_for('export_pdf')}" class="btn btn-success btn-sm">📄 Export PDF</a>
    <a href="{url_for('ml_metrics')}" class="btn btn-outline btn-sm">📊 ML Metrics</a>
  </div>

  <div class="card">
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;text-align:center">
      <div><div style="color:var(--muted);font-size:0.78rem">Patient</div><div style="font-weight:600">{name}</div></div>
      <div><div style="color:var(--muted);font-size:0.78rem">Age Group</div><div style="font-weight:600">{age_group}</div></div>
      <div><div style="color:var(--muted);font-size:0.78rem">Blood Group</div><div style="font-weight:600">{blood}</div></div>
      <div><div style="color:var(--muted);font-size:0.78rem">Date</div><div style="font-weight:600">{now[:10]}</div></div>
      <div><div style="color:var(--muted);font-size:0.78rem">Severity</div><div style="font-weight:600">{severity}/5</div></div>
      <div><div style="color:var(--muted);font-size:0.78rem">Duration</div><div style="font-weight:600">{duration}</div></div>
    </div>
  </div>

  <div class="card" style="text-align:center">
    <div class="section-title" style="justify-content:center">⚖ Body Mass Index</div>
    <div style="font-size:2.5rem;font-weight:700;color:{bmi_col};font-family:'DM Serif Display',serif">{bmi}</div>
    <div style="color:{bmi_col};font-weight:600;margin-top:0.25rem">{cat}</div>
    <div style="color:var(--muted);font-size:0.82rem;margin-top:0.25rem">Healthy range: 18.5 – 24.9</div>
  </div>

  <div class="card" style="border-color:var(--btn)">
    <h2 style="font-family:'DM Serif Display',serif;font-size:1.6rem;color:var(--accent)">🩺 {problem}</h2>
    {f'<p style="color:var(--muted);font-size:0.85rem;margin-top:0.4rem">Commonly affects: {detail.get("age_groups","")}</p>' if detail.get("age_groups") else ""}
  </div>

  {sections_html}

  <div class="card" style="text-align:center;background:linear-gradient(135deg,var(--card),var(--card2))">
    <div style="color:var(--success);font-size:1.2rem;font-weight:700;margin-bottom:0.75rem">✦ Analysis Complete!</div>
    <div style="display:flex;gap:0.75rem;justify-content:center;flex-wrap:wrap">
      <a href="{url_for('feedback')}" class="btn btn-success">⭐ Give Feedback</a>
      <a href="{url_for('appointments')}" class="btn">📅 Book Appointment</a>
      <a href="{url_for('history')}" class="btn btn-outline">📂 View History</a>
    </div>
  </div>
</div>"""
    return render_page(html, active_page="analysis")

# ── EXPORT PDF ──
@app.route("/export_pdf")
@login_required
def export_pdf():
    if not HAS_RL:
        return "reportlab not installed. Run: pip install reportlab", 400
    problem = session.get("problem","")
    name = session.get("user_name","N/A")
    detail_key = PROBLEM_MAP.get(problem,"")
    detail = HEALTH_DETAILS.get(detail_key,{})
    h = float(session.get("user_height") or 0)
    w = float(session.get("user_weight") or 0)
    bmi, cat = bmi_calc(w,h) if h and w else (0,"N/A")
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    styles = getSampleStyleSheet()
    ts = ParagraphStyle("t", parent=styles["Title"], textColor=colors.purple, spaceAfter=12)
    bs = ParagraphStyle("b", parent=styles["Normal"], fontSize=10, spaceAfter=5)
    story = [Paragraph("AI Women's Health Analysis Report", ts), Spacer(1,12),
             Paragraph(f"Patient: {name} | Problem: {problem} | Date: {now} | BMI: {bmi} ({cat})", bs), Spacer(1,8)]
    for section, key in [("CAUSES","causes"),("SYMPTOMS","symptoms"),("PRECAUTIONS","precautions"),
                         ("DIET PLAN","diet_plan"),("YOGA & EXERCISE","yoga_poses"),
                         ("MEDICINES","medicines"),("RECOMMENDED DOCTORS","doctors")]:
        if detail.get(key):
            story.append(Paragraph(f"<b>{section}</b>", bs))
            for item in detail[key]:
                story.append(Paragraph(f"  • {item}", bs))
    doc.build(story)
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name="health_report.pdf", mimetype="application/pdf")

# ── BMI ──
@app.route("/bmi", methods=["GET","POST"])
@login_required
def bmi():
    result = ""
    h_val = session.get("user_height","")
    w_val = session.get("user_weight","")
    if request.method == "POST":
        try:
            weight = float(request.form["weight"])
            height = float(request.form["height"])
            b, cat = bmi_calc(weight, height)
            col = {"Underweight":"var(--info)","Normal":"var(--success)","Overweight":"var(--warning)","Obese":"var(--danger)"}.get(cat,"var(--accent)")
            tips = {"Underweight":"Increase caloric intake with nutritious foods & strength training.","Normal":"Excellent! Maintain your healthy lifestyle. 🎉","Overweight":"Add 30 min cardio daily & reduce processed foods.","Obese":"Consult a nutritionist and doctor for a structured plan."}
            pct = min(int(b * 2.5), 100)
            result = f"""
<div class="card" style="text-align:center;margin-top:1rem">
  <div style="font-size:3rem;font-weight:700;color:{col};font-family:'DM Serif Display',serif">{b}</div>
  <div style="font-size:1.3rem;font-weight:600;color:{col}">{cat}</div>
  <div style="margin:1rem 0">
    <div style="display:flex;justify-content:space-between;font-size:0.78rem;color:var(--muted);margin-bottom:0.3rem">
      <span>10</span><span style="color:var(--info)">Underweight &lt;18.5</span><span style="color:var(--success)">Normal 18.5–25</span><span style="color:var(--warning)">Over 25–30</span><span style="color:var(--danger)">Obese 30+</span><span>40</span>
    </div>
    <div class="metric-bar-bg" style="height:16px">
      <div class="metric-bar-fill" style="width:{pct}%;background:{col}"></div>
    </div>
  </div>
  <p style="color:var(--muted)">{tips.get(cat,"")}</p>
</div>"""
            h_val, w_val = height, weight
        except: result = '<div class="alert alert-danger">Please enter valid numbers.</div>'

    html = f"""
<div class="page-wrap" style="max-width:600px">
  <div class="page-header"><h1>⚖ BMI Calculator</h1><p>Calculate your Body Mass Index</p></div>
  <div class="card">
    <form method="POST">
      <div class="two-col">
        <div class="form-group"><label>Weight (kg)</label><input type="number" step="0.1" name="weight" value="{w_val}" required></div>
        <div class="form-group"><label>Height (cm)</label><input type="number" step="0.1" name="height" value="{h_val}" required></div>
      </div>
      <button class="btn" type="submit" style="width:100%;justify-content:center">Calculate BMI</button>
    </form>
  </div>
  {result}
</div>"""
    return render_page(html, active_page="bmi")

# ── CYCLE TRACKER ──
@app.route("/cycle", methods=["GET","POST"])
@login_required
def cycle():
    email = session["user_email"]
    msg = ""
    if request.method == "POST":
        try:
            length = int(request.form.get("cycle_length","28"))
        except: length = 28
        db = get_db()
        db.execute("INSERT INTO cycle_tracker(user_email,period_start,period_end,cycle_length,notes) VALUES(?,?,?,?,?)",
            (email, request.form["period_start"], request.form["period_end"], length, request.form.get("notes","")))
        db.commit(); db.close()
        msg = '<div class="alert alert-success">✅ Cycle log saved!</div>'

    db = get_db()
    rows = db.execute("SELECT * FROM cycle_tracker WHERE user_email=? ORDER BY id DESC LIMIT 5", (email,)).fetchall()
    db.close()

    pred_html = ""
    if rows:
        try:
            latest = rows[0]
            last = datetime.datetime.strptime(latest["period_start"], "%Y-%m-%d")
            cl = latest["cycle_length"] or 28
            nxt = last + datetime.timedelta(days=cl)
            ov = last + datetime.timedelta(days=cl-14)
            fs = ov - datetime.timedelta(days=2)
            fe = ov + datetime.timedelta(days=2)
            pred_html = f"""
<div class="card">
  <div class="section-title">📅 Cycle Predictions</div>
  <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:0.75rem">
    <div style="background:var(--card2);padding:0.9rem;border-radius:10px;border:1px solid var(--border)">
      <div style="color:var(--muted);font-size:0.78rem">Last Period Start</div>
      <div style="font-weight:600">{latest["period_start"]}</div>
    </div>
    <div style="background:var(--card2);padding:0.9rem;border-radius:10px;border:1px solid var(--border)">
      <div style="color:var(--muted);font-size:0.78rem">Cycle Length</div>
      <div style="font-weight:600">{cl} days</div>
    </div>
    <div style="background:rgba(192,132,252,0.1);padding:0.9rem;border-radius:10px;border:1px solid rgba(192,132,252,0.3)">
      <div style="color:var(--muted);font-size:0.78rem">📅 Next Period</div>
      <div style="font-weight:700;color:var(--accent)">{nxt.strftime('%Y-%m-%d')}</div>
    </div>
    <div style="background:rgba(74,222,128,0.1);padding:0.9rem;border-radius:10px;border:1px solid rgba(74,222,128,0.3)">
      <div style="color:var(--muted);font-size:0.78rem">🥚 Ovulation</div>
      <div style="font-weight:700;color:var(--success)">{ov.strftime('%Y-%m-%d')}</div>
    </div>
    <div style="background:rgba(250,204,21,0.1);padding:0.9rem;border-radius:10px;border:1px solid rgba(250,204,21,0.3);grid-column:span 2">
      <div style="color:var(--muted);font-size:0.78rem">💫 Fertile Window</div>
      <div style="font-weight:700;color:var(--warning)">{fs.strftime('%d %b')} – {fe.strftime('%d %b')}</div>
    </div>
  </div>
</div>"""
        except: pred_html = '<div class="alert alert-warning">Use YYYY-MM-DD format for dates.</div>'

    logs_html = ""
    if rows:
        rows_html = "".join(f'<tr><td>{r["period_start"]}</td><td>{r["period_end"]}</td><td>{r["cycle_length"]} days</td><td style="color:var(--muted)">{r["notes"] or "-"}</td></tr>' for r in rows)
        logs_html = f"""
<div class="card">
  <div class="section-title">📝 Recent Logs</div>
  <table><tr><th>Start</th><th>End</th><th>Length</th><th>Notes</th></tr>{rows_html}</table>
</div>"""

    html = f"""
<div class="page-wrap">
  <div class="page-header"><h1>🌸 Menstrual Cycle Tracker</h1><p>Log and predict your menstrual cycle</p></div>
  {msg}
  <div class="two-col">
    <div>
      <div class="card">
        <div class="section-title">➕ Add Cycle Log</div>
        <form method="POST">
          <div class="form-group"><label>Period Start (YYYY-MM-DD)</label><input name="period_start" placeholder="2024-03-01" required></div>
          <div class="form-group"><label>Period End (YYYY-MM-DD)</label><input name="period_end" placeholder="2024-03-05" required></div>
          <div class="form-group"><label>Cycle Length (days)</label><input type="number" name="cycle_length" value="28" min="20" max="45"></div>
          <div class="form-group"><label>Notes (optional)</label><input name="notes" placeholder="Any symptoms..."></div>
          <button class="btn" style="width:100%;justify-content:center">💾 Save Log</button>
        </form>
      </div>
    </div>
    <div>
      {pred_html}
      {logs_html}
    </div>
  </div>
</div>"""
    return render_page(html, active_page="cycle")

# ── APPOINTMENTS ──
@app.route("/appointments", methods=["GET","POST"])
@login_required
def appointments():
    email = session["user_email"]
    msg = ""
    if request.method == "POST":
        db = get_db()
        db.execute("INSERT INTO appointments(user_email,doctor_name,specialty,date,time,notes,status) VALUES(?,?,?,?,?,?,?)",
            (email, request.form["doctor_name"], request.form["specialty"],
             request.form["date"], request.form["time"], request.form.get("notes",""), "Scheduled"))
        db.commit(); db.close()
        msg = '<div class="alert alert-success">✅ Appointment booked!</div>'

    db = get_db()
    appts = db.execute("SELECT * FROM appointments WHERE user_email=? ORDER BY date DESC", (email,)).fetchall()
    db.close()

    rows_html = "".join(f'<tr><td>{a["date"]} {a["time"]}</td><td>Dr. {a["doctor_name"]}</td><td>{a["specialty"]}</td><td>{a["notes"] or "-"}</td><td><span class="badge badge-success">{a["status"]}</span></td></tr>' for a in appts) or "<tr><td colspan='5' style='color:var(--muted);text-align:center'>No appointments yet.</td></tr>"

    html = f"""
<div class="page-wrap">
  <div class="page-header"><h1>📅 Appointment Booking</h1><p>Schedule and manage your doctor appointments</p></div>
  {msg}
  <div class="two-col">
    <div class="card">
      <div class="section-title">➕ Book Appointment</div>
      <form method="POST">
        <div class="form-group"><label>Doctor Name</label><input name="doctor_name" required></div>
        <div class="form-group"><label>Specialty</label>
          <select name="specialty">
            <option>Gynaecologist</option><option>Nutritionist</option><option>Dermatologist</option>
            <option>Physiotherapist</option><option>Endocrinologist</option><option>Psychiatrist</option>
            <option>General Physician</option><option>Neurologist</option><option>Oncologist</option>
          </select>
        </div>
        <div class="form-group"><label>Date (YYYY-MM-DD)</label><input name="date" placeholder="2024-06-15" required></div>
        <div class="form-group"><label>Time</label><input name="time" placeholder="10:30 AM" required></div>
        <div class="form-group"><label>Notes</label><textarea name="notes" style="min-height:70px" placeholder="Reason for visit..."></textarea></div>
        <button class="btn" style="width:100%;justify-content:center">📌 Book Appointment</button>
      </form>
    </div>
    <div class="card">
      <div class="section-title">Your Appointments</div>
      <div style="overflow-x:auto">
        <table><tr><th>Date/Time</th><th>Doctor</th><th>Specialty</th><th>Notes</th><th>Status</th></tr>{rows_html}</table>
      </div>
    </div>
  </div>
</div>"""
    return render_page(html, active_page="appointments")

# ── HISTORY ──
@app.route("/history")
@login_required
def history():
    email = session["user_email"]
    db = get_db()
    records = db.execute("SELECT recorded_at,problem,bmi,bmi_category FROM health_records WHERE user_email=? ORDER BY id DESC", (email,)).fetchall()
    db.close()

    rows_html = "".join(f"""<tr>
      <td>{r["recorded_at"]}</td>
      <td>{r["problem"]}</td>
      <td>{r["bmi"]}</td>
      <td><span class="badge {'badge-success' if r['bmi_category']=='Normal' else 'badge-warning' if r['bmi_category']=='Overweight' else 'badge-danger' if r['bmi_category']=='Obese' else 'badge-info'}">{r["bmi_category"]}</span></td>
    </tr>""" for r in records) or "<tr><td colspan='4' style='color:var(--muted);text-align:center'>No health records yet.</td></tr>"

    html = f"""
<div class="page-wrap">
  <div class="page-header"><h1>📂 Health History</h1><p>Your complete health analysis records</p></div>
  <div style="display:flex;gap:0.75rem;margin-bottom:1rem;flex-wrap:wrap">
    <a href="{url_for('export_csv')}" class="btn btn-success btn-sm">⬇ Export CSV</a>
    <a href="{url_for('analysis')}" class="btn btn-sm">🔬 New Analysis</a>
  </div>
  <div class="card">
    <div class="section-title">🩺 Health Records</div>
    <div style="overflow-x:auto">
      <table>
        <tr><th>Date</th><th>Problem</th><th>BMI</th><th>Category</th></tr>
        {rows_html}
      </table>
    </div>
  </div>
</div>"""
    return render_page(html, active_page="history")

# ── EXPORT CSV ──
@app.route("/export_csv")
@login_required
def export_csv():
    if not HAS_PANDAS:
        return "pandas not installed. Run: pip install pandas", 400
    email = session["user_email"]
    db = get_db()
    rows = db.execute("SELECT * FROM health_records WHERE user_email=?", (email,)).fetchall()
    db.close()
    data = [dict(r) for r in rows]
    df = pd.DataFrame(data)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return send_file(io.BytesIO(buf.getvalue().encode()), as_attachment=True,
                     download_name="health_data.csv", mimetype="text/csv")

# ── ML METRICS ──
@app.route("/ml_metrics")
@login_required
def ml_metrics():
    if not HAS_SK:
        html = """<div class="page-wrap" style="max-width:600px">
  <div class="page-header"><h1>📊 ML Model Metrics</h1></div>
  <div class="alert alert-warning">⚠ scikit-learn not installed.<br>Run: <code>pip install scikit-learn numpy</code></div>
</div>"""
        return render_page(html, active_page="ml_metrics")

    if "error" in _ML_METRICS:
        html = f'<div class="page-wrap"><div class="alert alert-danger">Metrics error: {_ML_METRICS["error"]}</div></div>'
        return render_page(html, active_page="ml_metrics")

    acc  = _ML_METRICS.get("accuracy", 0)
    prec = _ML_METRICS.get("precision",0)
    rec  = _ML_METRICS.get("recall",   0)
    f1   = _ML_METRICS.get("f1",       0)
    grade, gcol = ("A – Excellent","var(--success)") if acc >= 90 else \
                  ("B – Good","var(--info)") if acc >= 75 else \
                  ("C – Moderate","var(--warning)") if acc >= 60 else \
                  ("D – Needs Work","var(--danger)")

    def bar_row(name, val, col, desc):
        return f"""
<div style="display:grid;grid-template-columns:150px 1fr 280px;gap:1rem;align-items:center;margin-bottom:1rem">
  <div>
    <div style="font-weight:600;color:{col}">{name}</div>
    <div style="font-size:1.5rem;font-weight:700;font-family:'DM Serif Display',serif">{val}%</div>
  </div>
  <div>
    <div class="metric-bar-bg">
      <div class="metric-bar-fill" style="width:{val}%;background:{col}"></div>
    </div>
  </div>
  <div style="font-size:0.82rem;color:var(--muted)">{desc}</div>
</div>"""

    metrics_rows = (
        bar_row("Accuracy",  acc,  "var(--success)", "Overall correct predictions out of all predictions made.") +
        bar_row("Precision", prec, "var(--info)",    "Of predicted positives, how many were actually positive (weighted avg).") +
        bar_row("Recall",    rec,  "var(--accent)",  "Of actual positives, how many did the model correctly identify (weighted avg).") +
        bar_row("F1 Score",  f1,   "var(--warning)", "Harmonic mean of Precision and Recall – balanced performance measure.")
    )

    report = _ML_METRICS.get("report","No report.")
    labels = _ML_METRICS.get("labels",[])
    cm = _ML_METRICS.get("cm",[])

    # Build CM table
    cm_html = ""
    if labels and cm:
        header = "<tr><th>Actual \\ Pred</th>" + "".join(f"<th style='font-size:0.7rem;writing-mode:vertical-rl'>{l}</th>" for l in labels) + "</tr>"
        rows_cm = ""
        for i, lbl in enumerate(labels):
            cells = ""
            for j in range(len(labels)):
                v = cm[i][j]
                bg = "rgba(192,132,252,0.5)" if i==j and v>0 else ("rgba(248,113,113,0.3)" if v>0 else "transparent")
                cells += f'<td style="text-align:center;background:{bg};font-size:0.8rem">{v}</td>'
            rows_cm += f"<tr><td style='font-size:0.78rem;white-space:nowrap'>{lbl}</td>{cells}</tr>"
        cm_html = f'<div style="overflow:auto"><table style="font-size:0.8rem">{header}{rows_cm}</table></div>'

    guide_items = [
        ("Accuracy","% of all test samples correctly classified. Simple and intuitive."),
        ("Precision","When predicting class X, how often is it actually X? High = fewer false positives."),
        ("Recall","Of all real class-X samples, how many did the model catch? High = fewer false negatives."),
        ("F1 Score","Harmonic mean of P & R. F1 = 2×(P×R)/(P+R). Use when you need a single balanced metric."),
        ("Confusion Matrix","Cell (i,j) = how many times actual class i was predicted as j. Perfect = diagonal only."),
        ("Cross-Validation","Data split into N folds. Train on N-1, test on 1. Repeated N times. More reliable than single split."),
    ]
    guide_html = "".join(f'<div style="display:grid;grid-template-columns:160px 1fr;gap:0.5rem;padding:0.5rem 0;border-bottom:1px solid var(--border)"><div style="font-weight:600;color:var(--accent2)">{k}</div><div style="font-size:0.87rem;color:var(--muted)">{v}</div></div>' for k,v in guide_items)

    html = f"""
<div class="page-wrap">
  <div class="page-header"><h1>📊 ML Model Metrics</h1><p>Algorithm performance evaluation – Multinomial Naive Bayes + TF-IDF</p></div>
  <div style="display:flex;gap:0.75rem;margin-bottom:1.5rem;flex-wrap:wrap">
    <a href="{url_for('dashboard')}" class="btn btn-outline btn-sm">← Dashboard</a>
    <a href="{url_for('export_ml_report')}" class="btn btn-success btn-sm">💾 Export Report</a>
  </div>

  <div class="card">
    <div class="section-title">🤖 Algorithm Information</div>
    <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:0.5rem">
      {"".join(f'<div style="background:var(--card2);padding:0.75rem;border-radius:8px;border:1px solid var(--border)"><div style="color:var(--muted);font-size:0.78rem">{k}</div><div style="font-weight:600;font-size:0.9rem">{v}</div></div>' for k,v in [("Algorithm",_ML_METRICS.get("algo","N/A")),("Dataset Size",f"{_ML_METRICS.get('n_samples',0)} samples"),("Classes",f"{_ML_METRICS.get('n_classes',0)} health conditions"),("CV Strategy",f"{_ML_METRICS.get('cv_folds',0)}-Fold Stratified Cross-Validation"),("Features","TF-IDF Vectorizer (unigrams + bigrams)"),("Evaluation","Cross-val predictions vs ground-truth")])}
    </div>
  </div>

  <div class="card">
    <div class="section-title">📈 Key Performance Metrics</div>
    {metrics_rows}
    <div style="display:flex;align-items:center;gap:0.5rem;padding-top:0.5rem;border-top:1px solid var(--border)">
      <span style="color:var(--muted);font-size:0.88rem">Overall Grade:</span>
      <span style="font-weight:700;font-size:1rem;color:{gcol}">{grade}</span>
    </div>
  </div>

  <div class="card">
    <div class="section-title">🔲 Confusion Matrix</div>
    <p style="color:var(--muted);font-size:0.82rem;margin-bottom:0.75rem">Rows = Actual | Columns = Predicted | Purple diagonal = Correct predictions</p>
    {cm_html if cm_html else '<p style="color:var(--muted)">No confusion matrix data.</p>'}
  </div>

  <div class="card">
    <div class="section-title">📝 Per-Class Classification Report</div>
    <pre class="mono" style="background:var(--entry);padding:1rem;border-radius:8px;overflow-x:auto;white-space:pre;color:var(--text);line-height:1.6">{report}</pre>
  </div>

  <div class="card">
    <div class="section-title">📖 How to Interpret These Metrics</div>
    {guide_html}
  </div>
</div>"""
    return render_page(html, active_page="ml_metrics")

# ── EXPORT ML REPORT ──
@app.route("/export_ml_report")
@login_required
def export_ml_report():
    if not HAS_SK or "error" in _ML_METRICS:
        return "Metrics not available.", 400
    lines = [
        "="*60, "  AI WOMEN'S HEALTH – ML MODEL METRICS REPORT",
        f"  Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", "="*60, "",
        f"Algorithm     : {_ML_METRICS.get('algo','N/A')}",
        f"Dataset Size  : {_ML_METRICS.get('n_samples',0)} samples",
        f"Classes       : {_ML_METRICS.get('n_classes',0)} health conditions",
        f"CV Folds      : {_ML_METRICS.get('cv_folds',0)}-Fold Stratified CV", "",
        "── KEY METRICS ──────────────────────────────────",
        f"  Accuracy  : {_ML_METRICS.get('accuracy',0)}%",
        f"  Precision : {_ML_METRICS.get('precision',0)}%  (weighted avg)",
        f"  Recall    : {_ML_METRICS.get('recall',0)}%  (weighted avg)",
        f"  F1 Score  : {_ML_METRICS.get('f1',0)}%  (weighted avg)", "",
        "── CLASSIFICATION REPORT ────────────────────────",
        _ML_METRICS.get("report",""), "",
    ]
    content = "\n".join(lines)
    return send_file(io.BytesIO(content.encode()), as_attachment=True,
                     download_name="ml_metrics_report.txt", mimetype="text/plain")

# ── FEEDBACK ──
@app.route("/feedback", methods=["GET","POST"])
@login_required
def feedback():
    email = session["user_email"]
    msg = ""
    if request.method == "POST":
        text = request.form.get("text","").strip()
        rating = request.form.get("rating","0")
        if not text or rating == "0":
            msg = '<div class="alert alert-warning">Please write feedback and select a rating.</div>'
        else:
            db = get_db()
            db.execute("INSERT INTO feedback(user_email,text,rating,created_at) VALUES(?,?,?,?)",
                (email, text, int(rating), datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
            db.commit(); db.close()
            msg = '<div class="alert alert-success">💜 Thank you! Your feedback has been saved.</div>'

    db = get_db()
    row = db.execute("SELECT AVG(rating), COUNT(*) FROM feedback").fetchone()
    db.close()
    avg = round(row[0] or 0, 1)
    cnt = row[1]
    stars = "⭐"*int(avg) + "☆"*(5-int(avg))

    html = f"""
<div class="page-wrap" style="max-width:640px">
  <div class="page-header"><h1>⭐ Feedback</h1><p>STEP 5 of 5 – Share your experience</p></div>

  <div class="steps" style="margin-bottom:1.5rem">
    <div class="step done">1 Account</div>
    <div class="step done">2 Details</div>
    <div class="step done">3 Analysis</div>
    <div class="step done">4 Solutions</div>
    <div class="step active">5 Feedback</div>
  </div>

  {msg}

  <div class="card">
    <div class="section-title">Share Your Experience</div>
    <form method="POST" id="fbForm">
      <div class="form-group">
        <label>Your Experience</label>
        <textarea name="text" placeholder="Tell us how WellnessAI helped you..." required></textarea>
      </div>
      <div class="form-group">
        <label>Category</label>
        <select name="category">
          <option>General</option><option>Analysis Quality</option><option>UI Design</option>
          <option>Suggestions</option><option>Bug Report</option>
        </select>
      </div>
      <div class="form-group">
        <label>Rating</label>
        <div style="display:flex;gap:0.5rem;margin-top:0.3rem">
          {"".join(f'<label style="cursor:pointer;font-size:1.8rem" title="{i} star"><input type="radio" name="rating" value="{i}" style="display:none" required>{"⭐" if i<=3 else "☆"}</label>' for i in range(1,6))}
        </div>
      </div>
      <div class="form-group">
        <label>Would you recommend WellnessAI?</label>
        <div style="display:flex;gap:1rem">
          {"".join(f'<label style="display:flex;align-items:center;gap:0.3rem;cursor:pointer"><input type="radio" name="recommend" value="{o}" {"checked" if o=="Yes" else ""}> {o}</label>' for o in ["Yes","Maybe","No"])}
        </div>
      </div>
      <button class="btn btn-success" style="width:100%;justify-content:center" type="submit">✅ Submit Feedback</button>
    </form>
  </div>

  <div class="card" style="text-align:center">
    <div class="section-title" style="justify-content:center">📊 Community Ratings</div>
    <div style="font-size:1.5rem">{stars}</div>
    <div style="color:var(--success);font-weight:600;margin-top:0.25rem">{avg}/5  •  {cnt} reviews</div>
  </div>

  <div class="card" style="text-align:center;background:linear-gradient(135deg,var(--card),var(--card2))">
    <p style="color:var(--accent);font-weight:600;font-size:1.1rem;margin-bottom:0.25rem">Thank you for using AI Women's Health Analyser</p>
    <p style="color:var(--muted)">Your health journey matters. Come back anytime. 💜</p>
    <div style="display:flex;gap:0.75rem;justify-content:center;margin-top:1rem">
      <a href="{url_for('dashboard')}" class="btn btn-outline">🏠 Dashboard</a>
      <a href="{url_for('analysis')}" class="btn">🔬 New Analysis</a>
    </div>
  </div>
</div>

<script>
// Interactive star rating
const stars = document.querySelectorAll('[name="rating"]');
stars.forEach((s,i) => s.parentElement.addEventListener('mouseover', () => {
  document.querySelectorAll('[name="rating"]').forEach((el,j) => el.parentElement.textContent = (j<=i?'⭐':'☆'));
}));
</script>"""
    return render_page(html, active_page="feedback")

# ══════════════════════════════════════════════
if __name__ == "__main__":
    print("✦ AI Women's Health Analyser – Web App")
    print("  Starting server at http://127.0.0.1:5000")
    print("  Install deps: pip install flask scikit-learn pandas reportlab")
    app.run(debug=True, host="0.0.0.0", port=5000)
