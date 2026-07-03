# Diabetes Prediction

A simple machine learning project that trains a Decision Tree classifier to predict whether a person has diabetes using a tabular dataset. The work is implemented as a Jupyter/Colab notebook (Daibaties.ipynb).

## Project structure

- Daibaties.ipynb — Jupyter/Colab notebook with data loading, preprocessing, model training and evaluation.
- diabetes_dataset_50000_rows.csv — CSV dataset used by the notebook (expected path: `/content/diabetes_dataset_50000_rows.csv` when using Colab).

## Summary

The notebook:
- Loads the dataset and shows a sample of records.
- Drops an `id` column and separates features (`X`) from the target (`diabetes`).
- Splits the data into train and test sets (test_size=0.2, random_state=42).
- Applies standard scaling.
- Trains a DecisionTreeClassifier (max_depth=5).
- Evaluates the model and prints test accuracy.

Example reported result from the notebook: `Accuracy: 69.59%`. Your exact accuracy may vary depending on preprocessing, train/test split, scaling order, and random seeds.

## Requirements

Use Python 3.8+ and install the following packages:

- pandas
- numpy
- scikit-learn
- jupyter (if running locally)

Install with pip:

pip install pandas numpy scikit-learn jupyter

## How to run

Option A — Run in Google Colab (recommended)
1. Open the notebook in Colab:
   https://colab.research.google.com/github/arbhatti0041-max/Diabetes-Prediction/blob/main/Daibaties.ipynb
2. Make sure `diabetes_dataset_50000_rows.csv` is available in the Colab environment (you can upload it to Colab or mount Google Drive).
3. Run cells sequentially.

Option B — Run locally
1. Clone this repo:
   git clone https://github.com/arbhatti0041-max/Diabetes-Prediction.git
2. Install dependencies (see Requirements).
3. Start Jupyter:
   jupyter notebook
4. Open `Daibaties.ipynb`.
5. Ensure `diabetes_dataset_50000_rows.csv` is in the same directory or update `file_path` in the notebook to point to the dataset.
6. Run the notebook cells.

## Notes on the code (common issues & fixes)

- Typo fix: The first import cell in the notebook currently begins with `gimport pandas as pd` — remove the leading `g`.
- Scaling order: The notebook currently calls `scaled.fit_transform(X_test)` then `scaled.transform(X_train)`. Correct sequence should be:
  - scaled = StandardScaler()
  - X_train_scaled = scaled.fit_transform(X_train)
  - X_test_scaled  = scaled.transform(X_test)
  Then use the scaled arrays for training and prediction.
- Ensure you use the scaled training set for `model.fit(...)` and the scaled test set for predictions.

## Suggestions for improvement

- Try different models: RandomForest, LogisticRegression, SVM, GradientBoosting.
- Hyperparameter tuning with GridSearchCV or RandomizedSearchCV.
- Cross-validation for more robust evaluation.
- Feature engineering: create derived features, handle outliers, examine feature importance.
- Address class imbalance if present (oversampling, undersampling, class weights).
- Add model persistence (joblib / pickle) to save and load trained models.
- Add tests and CI to ensure notebook reproducibility.

## Reproducibility checklist

- Fix the `gimport` typo.
- Correct the StandardScaler usage as noted above.
- Set random seeds for model and train_test_split where applicable.
- Save model training and evaluation logs.

## License

Add your preferred license here (e.g., MIT). If you want, I can add an appropriate LICENSE file.

## Author / Contact

Repository owner: arbhatti0041-max (GitHub)
Contact: add your email or other contact information here.
