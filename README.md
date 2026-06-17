# 🏦 Bank Customer Churn Intelligence

A premium, interactive **Streamlit dashboard** for predicting bank customer churn using Gradient Boosting models. Features real-time predictions, comprehensive EDA visualizations, and a sleek dual-theme (dark/light) UI with glassmorphism design.

---

## ✨ Features

### 🎯 Real-Time Churn Predictor
- **Animated circular gauge** displaying churn probability
- Color-coded risk badges — *Low Risk*, *At Risk*, *High Risk*
- Model confidence scoring with descriptive feedback
- Stay vs. churn split bar visualization
- Detailed customer profile summary panel

### 📊 Exploratory Data Analysis
- Churn distribution & churn rate by number of products
- Demographic breakdowns (country & gender)
- Numerical feature distributions with KDE curves
- Correlation heatmap across all encoded features
- Feature importance ranking per selected model
- **ROC Curve** and **Precision-Recall Curve** model comparisons
- Side-by-side **Confusion Matrix** comparison

### 📋 Data Explorer
- Interactive filters — churn status, country, age range
- Full dataset table with formatted currencies and status icons
- Summary statistics (mean, std, min, median, max) for filtered data

### 🎨 Design
- **Dual theme** — Dark Mode (🌙) and Light Mode (☀️) with sidebar toggle
- **Glassmorphism** cards with backdrop blur and subtle glow borders
- Smooth `fadeSlideIn` animations on all cards and charts
- Gradient accent title, pill-shaped tabs, and custom scrollbars
- **Matplotlib charts** automatically adapt to the selected theme
- Consistent indigo/violet color palette across all visualizations

---

## 🧠 Models

| Model | Description |
|-------|-------------|
| **Base Model** | Gradient Boosting classifier trained on the original dataset |
| **SMOTE Optimized** | Gradient Boosting classifier trained with SMOTE oversampling to handle class imbalance |

Both models are pre-trained and stored as `.pkl` files. The dashboard compares their performance side-by-side on ROC, PR, and confusion matrix charts.

### Input Features

| Feature | Type |
|---------|------|
| Credit Score | Numerical (350–850) |
| Age | Numerical (18–92) |
| Tenure | Numerical (0–10 years) |
| Balance | Numerical ($0–$250,000) |
| Number of Products | Categorical (1–4) |
| Has Credit Card | Binary (Yes/No) |
| Active Member | Binary (Yes/No) |
| Estimated Salary | Numerical ($0–$200,000) |
| Country | Categorical (France/Germany/Spain) |
| Gender | Categorical (Female/Male) |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- pip

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/churn-main.git
   cd churn-main
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv .venv

   # Windows
   .venv\Scripts\activate

   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the dashboard**
   ```bash
   streamlit run app.py
   ```

5. Open your browser at **http://localhost:8501**

---

## 📁 Project Structure

```
churn-main/
├── app.py                # Streamlit dashboard (main application)
├── requirements.txt      # Python dependencies
├── churn.csv             # Dataset — 10,000 bank customers
├── base.pkl              # Pre-trained Base Gradient Boosting model
├── smote.pkl             # Pre-trained SMOTE Gradient Boosting model
├── scaler.pkl            # Fitted StandardScaler for feature normalization
├── AOL_ML.ipynb          # Jupyter notebook with full ML pipeline & EDA
├── bank_churn_predictor.html  # Exported notebook (HTML)
└── README.md
```

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Interactive web dashboard framework |
| `pandas` | Data manipulation and analysis |
| `numpy` | Numerical computing |
| `matplotlib` | Static chart rendering |
| `seaborn` | Statistical data visualization |
| `scikit-learn` (1.6.1) | ML models, metrics, and preprocessing |

---

## 📸 Screenshots

> Run the app and toggle between 🌙 **Dark Mode** and ☀️ **Light Mode** using the sidebar switch.

---

## 📝 License

This project is for educational purposes.
