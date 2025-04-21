# 📈 One-Step-Ahead Option Pricing & Explainability (TSLA)

This is a Streamlit app that predicts the **next-day price of a European call option** on **TSLA** using:

- A **CNN-Attention MLP**
- A structured **Informer + Kolmogorov–Arnold Network (KAFIN)**
- A benchmark **GARCH + Black-Scholes** model

The app also provides **PDP (Partial Dependence Plots)** and **SHAP feature importances** to help interpret model behavior.

---

## 🧠 Project Motivation

The goal is to answer:
> "If I hold or buy an option today, is it worth keeping until tomorrow?"

Using 20 days of real TSLA data from `yfinance`, the app predicts the **next-day price** of a user-defined option contract. It computes **realized volatility** as a proxy for implied volatility and supports three types of models for side-by-side evaluation.

---

## 🚀 Running the App

### ✅ Step 1: Clone the repo
git clone https://github.com/ryan9dai/XAI_Project_App.git
cd XAI_Project_App
### ✅ Step 2: Install dependencies
Recommended: Use a virtual environment.

python -m venv XAIvenv

source XAIvenv/bin/activate  # On Windows: XAIvenv\Scripts\activate

pip install torch yfinance numpy pandas matplotlib shap scipy arch

### ✅ Step 3: Run the app
streamlit run app.py
Streamlit will open the app in your browser automatically. If not, go to http://localhost:8501.

## 📊 Example Input Parameters
After fetching the most recent 20 days of TSLA data, you can specify any in-the-money or at-the-money call option (as that's what the model was trained on). Here are a few valid examples:

### ✅ Example 1: Slightly In-The-Money, Short-Term
Strike: 235

Time to Maturity: 0.1

Risk-Free Rate: 0.015

Expected behavior: price close to intrinsic value, relatively low sensitivity to volatility.

### ✅ Example 2: At-The-Money, Medium-Term
Strike: 240

Time to Maturity: 0.25

Risk-Free Rate: 0.015

Expected behavior: highly sensitive to implied volatility and time to maturity. Great for SHAP/PDP insights.

### ✅ Example 3: Deep In-The-Money, Medium-Term
Strike: 200

Time to Maturity: 0.3

Risk-Free Rate: 0.015

Expected behavior: model should price it close to S − K, with minimal time value.

### 📌 Notes
Realized Volatility is used as a proxy for IV at inference.

Only call options are supported (European-style).

The app does not support historical IV surfaces, so inferred volatility is a simplified estimate.

Models were trained on TSLA options from 2019–2022, using ATM and ITM calls only.

## 🧠 XAI Features
### ✅ MLP Model:
SHAP (Gradient × Input): Shows which features contributed most to price

PDP: Explore how changing 1 feature affects predicted price (e.g., strike, TTM, IV)

### ✅ Informer+KAFIN Model:
Structured pricing head inspired by Kolmogorov–Arnold decomposition

PDP available for interpretability

SHAP not available due to model structure

## 📬 Contact
For questions or suggestions, contact:
Ryan Dai at ryan9dai.us@gmail.com

## 🧠 Academic Disclaimer
This project is for educational purposes only and is not intended for real trading or financial advice. Please feel free to share this project, and you can include my name.

## Use of AI
This project was primarily guided by AI and a family member. It was entirely coded using ChatGPT and Claude, using many different models from each.