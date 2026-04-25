# 🌸 AI Women’s Health Analyzer

An intelligent web-based healthcare application designed to help women monitor, analyze, and manage their health using machine learning and data-driven insights.

This system combines **Flask (backend)**, **SQLite database**, and **Machine Learning (TF-IDF + Naive Bayes)** to provide personalized health analysis and record management.

---

## 📌 Project Overview

The **AI Women’s Health Analyzer** allows users to:

* Register and manage personal health profiles
* Input symptoms and receive AI-based predictions
* Track BMI and health records
* Store medical history securely
* Generate health reports (PDF support)
* Book and manage appointments

The application is built as a **full-stack web app** using Python and runs locally in a browser.

---

## 🚀 Features

* 🔐 User Authentication (Login/Register)
* 🧠 AI-based Symptom Analysis (Naive Bayes Model)
* 📊 BMI Calculation & Health Tracking
* 🗂️ Health Record Storage (SQLite Database)
* 📅 Appointment Management System
* 📄 PDF Report Generation (ReportLab)
* 🌐 Web Interface using HTML/CSS
* ⚡ Fast and lightweight Flask backend

---

## 🛠️ Technologies Used

* **Python**
* **Flask (Web Framework)**
* **SQLite (Database)**
* **Scikit-learn (Machine Learning)**
* **TF-IDF Vectorization**
* **Naive Bayes Algorithm**
* **Pandas & NumPy**
* **ReportLab (PDF Generation)**
* **HTML, CSS (Frontend)**

---

## 📁 Project Structure

```
AI-Women-s-Health-Analyzer/
│
├── app.py                     # Main Flask application
├── main.py                   # Entry point (optional UI runner)
├── index.html                # Frontend UI
├── womens_health_v2.db       # SQLite database
├── requirements.txt          # Dependencies
├── Procfile                  # Deployment config (Gunicorn)
└── README.md                 # Project documentation
```

---

## ⚙️ Installation & Setup (VS Code)

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/AI-Women-s-Health-Analyzer.git
cd AI-Women-s-Health-Analyzer
```

---

### 2️⃣ Open in VS Code

* Open **VS Code**
* Click **File → Open Folder**
* Select the project folder

---

### 3️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate it:

* **Windows:**

```bash
venv\Scripts\activate
```

* **Mac/Linux:**

```bash
source venv/bin/activate
```

---

### 4️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 5️⃣ Run the Application

```bash
python app.py
```

---

### 6️⃣ Open in Browser

After running, open:

```
http://127.0.0.1:5000/
```

---

## 🧠 How the AI Model Works

1. User enters symptoms
2. Text is processed using **TF-IDF Vectorization**
3. A **Naive Bayes classifier** predicts possible conditions
4. Results are stored in the database
5. Output is shown in the web interface

---

## 🗄️ Database Details

The project uses **SQLite (`womens_health_v2.db`)** with tables:

* **users** → Stores user information
* **health_records** → Stores health analysis results
* **appointments** → Stores appointment details

---

## 📄 PDF Report Feature

* Generates downloadable health reports
* Built using **ReportLab**
* Includes:

  * User details
  * BMI
  * Health predictions

---

## 📷 User Interface

* Clean and responsive UI using HTML & CSS
* Single-page styled interface (`index.html`)
* Connected dynamically with Flask backend

