# 📑 Support Ticket Intelligence System

🚀 **Machine Learning Task 2 (2026)**
🏢 **Organization:** Future Interns
👨‍💻 **Developer:** Anthony Kangogo

---

## 📌 Project Overview

In modern SaaS and financial service environments, handling customer support tickets manually creates delays and inefficiencies.

This project builds an **AI-powered Decision Support System** that:

* Reads unstructured customer complaints
* Classifies them into business categories
* Assigns a **priority level (High, Medium, Low)** instantly

✅ Designed to handle **1,000,000+ records** efficiently

---

## 🎯 Objective

Automate ticket classification and prioritization to:

* Reduce operational backlog
* Improve response time
* Enhance customer satisfaction

---

## ⚙️ System Architecture

### 🔹 Data & Scalability

* **Dataset Size:** 1,000,000 customer complaints
* **Memory Optimization:** Out-of-Core Learning (`chunksize=15000`)
* **Feature Engineering:** Hashing Vectorizer (**2¹⁸ features**)

### 🔹 Model

* **Algorithm:** Stochastic Gradient Descent (SGD)
* **Loss Function:** `log_loss`
* **Training:** Incremental Learning (`partial_fit`)

---

## 🧠 NLP Pipeline

* **Text Normalization:** Lowercasing
* **Noise Removal:** Regex cleaning (special characters, redactions)
* **Stopword Removal:** Eliminates filler words
* **Keyword Focus:** Retains high-impact terms (e.g., *fraud, foreclosure, harassment*)

---

## 📊 Business Logic: Ticket Prioritization

| Category         | Example Issues             | Priority  |
| ---------------- | -------------------------- | --------- |
| Debt Collection  | Harassment, disputed debts | 🔴 High   |
| Mortgage         | Foreclosure threats        | 🔴 High   |
| Credit Reporting | Identity theft             | 🟡 Medium |
| Cards            | Fraud, billing issues      | 🟡 Medium |
| Banking          | Account issues             | 🟢 Low    |
| Loans            | Loan terms inquiries       | 🟢 Low    |

---

## 📈 Model Performance

* ✅ **Accuracy:** 84%
* 📌 **F1 Scores:**

  * Credit Reporting → **0.91**
  * Mortgage → **0.86**
  * Banking → **0.80**

📊 The model performs well even when categories have overlapping vocabulary.

---

## 💼 Business Impact

✔️ **Reduced Backlog** – Automates ticket sorting
✔️ **Risk Detection** – Flags high-priority financial/legal issues instantly
✔️ **Efficiency Boost** – Support teams focus on solving, not sorting

---

## 🚀 Getting Started

### 1️⃣ Clone Repository

```bash
git clone https://github.com/A-kango/FUTURE_ML_02
cd FUTURE_ML_02
```

### 2️⃣ Install Dependencies

```bash
pip install pandas scikit-learn matplotlib seaborn joblib streamlit
```

### 3️⃣ Train Model

```bash
python ticket_classifier_model.py
```

### 4️⃣ Run Dashboard

```bash
streamlit run app.py
```

---

## 🧪 Example Test

**Input:**

```
I am being harassed by collectors for a debt I already paid!
```

**Output:**

```
Category: Debt Collection  
Priority: High
```

---

## 🔗 Repository Link

👉 https://github.com/A-kango/FUTURE_ML_02

---

## 🌟 Why This Project Stands Out

* Handles **Big Data (1M+ records)**
* Uses **production-level ML techniques**
* Demonstrates **real business impact**
* Ready for **deployment (Streamlit dashboard)**

---

## 📬 Contact

If you'd like to collaborate or discuss opportunities:
**Anthony Kangogo**

---
