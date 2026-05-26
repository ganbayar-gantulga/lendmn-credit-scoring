# LendMN Credit Scoring System

ML-powered credit scoring model for the Mongolian lending market.

## Project Structure

```
lendmn-credit-scoring/
├── data/
│   └── credit_data.csv          # Synthetic dataset (5,000 applicants)
├── models/
│   ├── best_model.joblib        # Trained model (Logistic Regression)
│   ├── label_encoder.joblib     # Employment type encoder
│   ├── feature_cols.joblib      # Feature column list
│   ├── metrics.json             # Evaluation results
│   ├── evaluation.png           # ROC / confusion matrix / feature importance
│   └── score_distribution.png  # Credit score distribution plot
├── src/
│   ├── generate_data.py         # Synthetic data generation
│   ├── train.py                 # ML training pipeline
│   └── api.py                  # FastAPI inference endpoint
└── README.md
```

## Model Performance

| Model               | AUC    | CV-AUC | Precision | Recall |
|---------------------|--------|--------|-----------|--------|
| Logistic Regression | 0.9939 | 0.9963 | 0.850     | 0.654  |
| XGBoost             | 0.9929 | 0.9905 | 0.900     | 0.692  |
| Random Forest       | 0.9787 | 0.9839 | 0.397     | 0.885  |

**Best model: Logistic Regression (AUC = 0.9939)**

## Credit Score Bands (300–850)

| Score Range | Band      | Монгол   | Action                   |
|-------------|-----------|----------|--------------------------|
| 750–850     | Excellent | Маш сайн | Автомат зөвшөөрөл        |
| 700–749     | Good      | Сайн     | Зөвшөөрөл                |
| 650–699     | Fair      | Дундаж   | Нэмэлт баримт шаардлагатай |
| 580–649     | Poor      | Дор      | Хязгаарлагдмал зээл      |
| 300–579     | Very Poor | Маш дор  | Татгалзах                |

## Features

| Feature                | Description               | Impact  |
|------------------------|---------------------------|---------|
| `age`                  | Нас                       | Negative risk |
| `employment_type`      | Ажлын төрөл               | Varies  |
| `monthly_income`       | Сарын орлого (₮)         | Negative risk |
| `employment_months`    | Ажилласан хугацаа         | Negative risk |
| `num_existing_loans`   | Одоогийн зээлийн тоо      | Positive risk |
| `credit_history_months`| Зээлийн түүх              | Negative risk |
| `previous_defaults`    | Хугацаа хэтрэлт           | **High positive risk** |
| `num_dependents`       | Тэжээлд авах хүний тоо   | Positive risk |
| `loan_amount`          | Зээлийн дүн               | Positive risk |
| `debt_to_income`       | DTI харьцаа               | Positive risk |
| `loan_to_income`       | LTI харьцаа               | Positive risk |

## Quick Start

```bash
# 1. Install dependencies
pip install scikit-learn pandas numpy matplotlib seaborn xgboost joblib fastapi uvicorn

# 2. Generate data
python src/generate_data.py

# 3. Train model
python src/train.py

# 4. Start API
uvicorn src.api:app --reload
```

## API Usage

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "age": 30,
    "employment_type": "private",
    "monthly_income": 1500000,
    "employment_months": 24,
    "num_existing_loans": 1,
    "credit_history_months": 36,
    "previous_defaults": 0,
    "num_dependents": 1,
    "loan_amount": 3000000
  }'
```

**Response:**
```json
{
  "credit_score": 848,
  "score_band": "Маш сайн",
  "default_probability": 0.0001,
  "recommendation": "Зээл олгохыг зөвлөж байна.",
  "risk_level": "LOW",
  "debt_to_income": 0.1,
  "loan_to_income": 0.167
}
```

## Portfolio Notes

- **Dataset**: 5,000 synthetic applicants, 2.6% default rate
- **Validation**: 5-fold stratified cross-validation
- **Score mapping**: Logistic probability → 300–850 FICO-style range
- **Production-ready**: FastAPI endpoint, joblib model persistence

---
*Developed as part of AI Engineering portfolio — Hohai University*
