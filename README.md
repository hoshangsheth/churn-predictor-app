# 🚀 Customer Churn Predictor (Deep Learning + Streamlit)

A production-ready Machine Learning web app that predicts whether a customer is likely to churn using a trained Artificial Neural Network.

---

## 📌 Overview

This project predicts customer churn based on financial, demographic, and account activity features.

It is built with:

* TensorFlow / Keras (Deep Learning)
* Scikit-learn (Preprocessing)
* Streamlit (Deployment UI)

---

## 🧠 Model Details

* Model: Artificial Neural Network (ANN)

* Input Features:

  * Credit Score
  * Geography
  * Gender
  * Age
  * Tenure
  * Balance
  * Number of Products
  * Credit Card Status
  * Active Member Status
  * Estimated Salary

* Output:

  * Churn Probability
  * Classification: Churn / No Churn

---

## 🗂 Project Structure

```
churn-predictor-app/
│
├── app/
│   └── churn_app.py
│
├── artifacts/
│   ├── model.h5
│   ├── scaler.pkl
│   ├── label_encoder_gender.pkl
│   └── onehot_encoder_geo.pkl
│
├── data/
│   └── Churn_Modelling.csv
│
├── notebooks/
│   ├── experiments.ipynb
│   └── prediction.ipynb
│
├── requirements.txt
├── README.md
├── .gitignore
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/churn-predictor-app.git
cd churn-predictor-app
```

---

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app/churn_app.py
```

App will run on:
http://localhost:8501

---

## 🔁 How to Fork

1. Click **Fork** on GitHub
2. Clone your fork
3. Create a new branch
4. Make changes and push

---

## 📊 Dataset

Bank Customer Churn dataset located in:

```
data/Churn_Modelling.csv
```

---

## 🧩 Future Improvements

* Add SHAP explainability
* Deploy on Streamlit Cloud / AWS
* Build API using FastAPI
* Optimize model performance

---

## 👨‍💻 Author

**Hoshang Sheth**

* Portfolio: [www.hoshang-portfolio-ai.vercel.app](http://www.hoshang-portfolio-ai.vercel.app)
* GitHub: [www.github.com/hoshangsheth](http://www.github.com/hoshangsheth)
* LinkedIn: [www.linkedin.com/in/hoshangsheth](http://www.linkedin.com/in/hoshangsheth)

---

## 📜 License

This project is licensed under the MIT License.

---

## ⭐ Support

If you found this useful, consider giving it a ⭐ on GitHub!
