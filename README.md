# Student Performance Prediction — Beginner ML Project

A complete beginner-friendly machine learning project that estimates a student's final score from study hours, attendance, previous score, and completed assignments.

> **Teaching note:** The included CSV is synthetic, illustrative data generated for this demo. It is not real student data, and this model is not suitable for grading or other high-stakes decisions.

## 1. Project structure

```text
student_performance_prediction/
├── app.py
├── train_model.py
├── requirements.txt
├── data/
│   └── student_performance.csv
└── README.md
```

`student_score_model.joblib` is created when you train the model.

## 2. Run locally

Install Python 3.10 or newer. Open a terminal in this project folder.

### Windows

```bash
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

Streamlit prints a local URL, usually `http://localhost:8501`. Open it in your browser.

## 3. Test the model

1. Run `python train_model.py`.
2. Confirm it prints MAE, RMSE, R² and saves `student_score_model.joblib`.
3. Run `streamlit run app.py`.
4. Try several combinations of inputs and confirm the output stays between 0 and 100.
5. Check that the app displays the disclaimer about illustrative data.

The exact evaluation metrics can vary slightly with library versions. A lower MAE/RMSE generally means smaller errors on the held-out test set. R² closer to 1 indicates a better fit on that test split, but metrics on synthetic data do not prove real-world usefulness.

## 4. Free/low-cost deployment option: Streamlit Community Cloud

Availability and free-tier terms can change. Check the provider's current terms before deployment.

1. Create a GitHub repository and upload `app.py`, `train_model.py`, `requirements.txt`, `README.md`, and the `data/` folder.
2. Locally run `python train_model.py` once to verify the project works. The app needs the generated `student_score_model.joblib`, so either:
   - **Recommended simple route:** also commit `student_score_model.joblib` to this teaching repository, or
   - modify the app to train/load the model as part of deployment.
3. Sign in to Streamlit Community Cloud with GitHub and create a new app.
4. Select your repository, branch, and `app.py` as the main file.
5. Deploy, then test the public URL with sample inputs.
6. If the repository is public, do not include private data, passwords, access keys, or real student records.

## 5. AWS cost-safe deployment notes

- You do not need AWS to teach the ML workflow; local deployment is free apart from your computer/internet.
- Your AWS credits may cover eligible charges, but they are not a guarantee that every service is free.
- Do not create SageMaker notebooks/endpoints, EC2 instances, NAT gateways, or load balancers without checking the current pricing and your account's Free Plan restrictions.
- If you use AWS, set up a billing budget/cost alert first where available, check the service and region pricing, and delete resources when finished. A budget alert is not always a hard spending cap.
- Never put AWS access keys in source code or a public GitHub repository.

## 6. How the ML workflow works

1. **Collect data:** a row represents one example student.
2. **Choose features (X):** study hours, attendance, previous score, assignments completed.
3. **Choose target (y):** final score.
4. **Split data:** 80% for training, 20% for testing.
5. **Scale features:** standardize feature ranges in a pipeline.
6. **Train:** Linear Regression learns a relationship between inputs and score.
7. **Evaluate:** MAE, RMSE, and R² compare predictions with held-out scores.
8. **Save:** `joblib` stores the trained pipeline.
9. **Deploy:** Streamlit provides a browser form for predictions.
10. **Monitor responsibly:** real-world use requires representative, consented data, privacy safeguards, validation, and bias checks.

## 7. Classroom explanation

**Problem:** Estimate a final score from learning-related inputs.

**ML type:** Supervised learning, regression.

**Why regression?** The target is a numeric value from 0 to 100, not a category.

**Linear Regression idea:** It learns coefficients for a formula such as:

`predicted_score = b0 + b1*hours + b2*attendance + b3*previous_score + b4*assignments`

The model learns the coefficients from training examples. The actual learned coefficients are produced during training.

**MAE:** average absolute difference between predicted and actual scores, measured in score points.

**RMSE:** square root of average squared error; larger mistakes are penalized more.

**R²:** compares the model to a baseline that predicts the test-set mean. It can be negative if the model performs worse than that baseline.

## 8. Suggested 20-minute lecture

- 0–3 min: Explain the problem and real-world motivation.
- 3–6 min: Explain dataset, features, and target.
- 6–10 min: Explain supervised learning and regression.
- 10–14 min: Run `train_model.py`; discuss train/test split and metrics.
- 14–17 min: Launch the Streamlit app and change the inputs.
- 17–20 min: Discuss limitations, privacy, fairness, and questions.

## 9. Viva questions

1. **What is Machine Learning?** A way for computers to learn patterns from data.
2. **What is supervised learning?** Learning from examples that include the correct target.
3. **Why regression?** Because final score is numeric.
4. **What are features?** Inputs used to make a prediction.
5. **What is the target?** The value the model learns to predict.
6. **Why split train and test data?** To evaluate on examples not used for fitting.
7. **What is MAE?** Average absolute prediction error.
8. **What is overfitting?** When a model fits training data well but generalizes poorly.
9. **What is deployment?** Making the model accessible through an application or service.
10. **Can this demo decide a student's future?** No. It uses synthetic data and is only for teaching.
