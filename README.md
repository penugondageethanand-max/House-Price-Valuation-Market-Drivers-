# 🏠 House Price Valuation & Market Drivers

An interactive **Business Intelligence dashboard** built with Python and Streamlit that transforms raw housing data into actionable real-estate insights. The application combines exploratory data analysis, machine learning–powered price prediction, and strategic risk/opportunity assessment into a single unified platform.

---

## 📌 Problem Statement

Real-estate stakeholders (buyers, sellers, agents, investors) lack a quick, data-driven way to **estimate property values** and understand which amenity and structural factors drive prices the most. This project solves that problem by providing:

1. **Executive KPI Overview** — market-level snapshot at a glance.
2. **Deep Property Analytics** — feature correlations, price drivers, and model diagnostics.
3. **Interactive Valuation Calculator** — enter any property's details and receive an instant ML-powered price estimate with confidence bounds, comparable listings, and strategic recommendations.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or later
- pip (Python package manager)

### Installation

```bash
# 1. Clone this repository
git clone <your-repo-url>
cd house-price-project

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Download the Dataset

Download **train.csv** from the Kaggle competition below and place it in the project root directory (same folder as `app.py`):

> **Dataset Link:** [https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)

### Run the Application

```bash
streamlit run app.py
```

The dashboard will open automatically in your default browser at `http://localhost:8501`.

---

## 🛠️ Tech Stack

| Layer       | Technology                       |
|-------------|----------------------------------|
| Frontend    | Streamlit, Plotly                |
| Backend     | Python, pandas, NumPy            |
| ML Model    | Gradient Boosting (scikit-learn) |
| Data Source  | Ames Housing Dataset (Kaggle)   |

---

## 📊 Key Features

| Page                    | What It Shows                                                              |
|-------------------------|----------------------------------------------------------------------------|
| **Executive Overview**  | KPI cards (median price, price/sqft, spread), price distribution, top neighbourhoods, quality analysis |
| **Property Analytics**  | Feature importance ranking, correlation matrix, actual vs predicted scatter, house-style & bathroom impact |
| **Valuation Calculator**| Interactive sliders → predicted price + 95% CI, comparable properties, risk / opportunity / action boxes |

---

## 📁 Project Structure

```
house-price-project/
├── app.py               # Single-file Streamlit application (code)
├── requirements.txt     # Python library dependencies
├── README.md            # This file — project overview
├── Project_Report.md    # Detailed project documentation (convert to .docx)
└── train.csv            # Ames Housing dataset (download from Kaggle)
```

---

## 🤖 Model Details

- **Algorithm:** Gradient Boosting Regressor (300 estimators, learning rate 0.05)
- **Validation:** 80/20 train-test split + 5-fold cross-validation
- **Top 5 Drivers:** Overall Quality, Living Area, Total Square Footage, Year Built, Garage Area
- **Metrics:** R² ≈ 0.91, MAE ≈ $16,500 (typical on this dataset)

---

## 📜 License

This project is created for educational purposes as part of the BharatCares / IBM SkillsBuild Data Analytics Internship.

---

## 👤 Author

**Your Name**
- Institution: Your Institution Name
- Branch: Data Analytics
