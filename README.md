# Car Price Prediction / Car Valuation Tool

A Streamlit app that estimates the market price of a used car from its features (brand, year, mileage, horsepower, engine, transmission, colors, accident history...), using an ElasticNet model trained on a used-car dataset from Kaggle.

## Table of Contents

- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Data Pipeline](#data-pipeline-notebook-summary)
- [The App](#the-app-streamlit_carspy)
- [Requirements](#requirements)
- [How to Run the App](#how-to-run-the-app)
- [Retraining the Model](#retraining-the-model)
- [Known Limitations](#known-limitations)
- [License](#license)

## Dataset

**Source:** [Used Car Price Prediction Dataset](https://www.kaggle.com/datasets/taeefnajib/used-car-price-prediction-dataset) (Kaggle)

**Included file:** `used_cars.csv`

**Original columns:** `brand`, `model`, `model_year`, `milage`, `fuel_type`, `engine`, `transmission`, `ext_col`, `int_col`, `accident`, `clean_title`, `price`

## Repository Structure

| File | Description |
|---|---|
| `used_cars.csv` | Original (raw) dataset |
| `test_carDep.ipynb` | Notebook with the full pipeline: cleaning, feature engineering, encoding, training and evaluation |
| `streamlit_cars.py` | Web app (Streamlit) |
| `pre_norm_dummied_df.parquet` | Intermediate dataset (post-dummies, pre-scaling) used for the app's plots |
| `elastic_net_model.pkl` | Trained ElasticNet model |
| `StandardScaler.pkl` | Scaler fitted on the numeric columns of the training set |
| `Target_encoder.pkl` | TargetEncoder (`category_encoders`) fitted on the `brand` column |
| `sorted_coef_dict.pkl` | Dictionary `{column: coefficient}` from the model, sorted, used for the feature-importance plot |

## Data Pipeline (notebook summary)

1. Load data and handle missing values
2. Fix column dtypes
3. Handle outliers (e.g. `milage`)
4. Simplify complex columns:
   - `engine` → `HP`, `Cylinders`, `Engine_Size` (L)
   - `transmission` → `Automatic` / `CVT` / `Manual` / `Other`
5. Low-cardinality categorical columns → One-Hot encoding (dummies)
6. High-cardinality categorical column (`brand`) → Target Encoding (`smoothing=50`, `min_samples_leaf=20`)
7. Train/test split (done **before** fitting encoders/scaler, to avoid data leakage)
8. Scaling (`StandardScaler`) of the non-binary numeric columns
9. Train an `ElasticNet` model with hyperparameter search (`alpha`, `l1_ratio`) via K-Fold cross-validation
10. Evaluation: MAE, MSE, RMSE, R²
11. Predict new samples and generate plots

> The model target is `log1p(price)`; predictions are converted back with `np.expm1()` before being shown to the user.

## The App (`streamlit_cars.py`)

**Features:**

- A form to enter the car's details
- On submit:
  - Builds the input row, encoding categorical variables the same way as during training (dummies + target encoding for the brand)
  - Scales the numeric columns with the saved `StandardScaler`
  - Predicts the price with the ElasticNet model
  - Shows the estimated value with a color code (🔴 red / 🟡 yellow / 🟢 green depending on the price range)
  - Shows similar-priced cars from the dataset
  - Generates two plots:
    - Price (log) vs. Year for the selected brand, comparing historical data against the submitted car
    - Feature importance based on the model's coefficients

## Requirements

Python 3.12 (recommended; tested with a conda environment)

**Main libraries:**

- `streamlit`
- `pandas`
- `numpy`
- `scikit-learn`
- `category_encoders`
- `joblib`
- `matplotlib`
- `pyarrow` (to read the `.parquet` file)

### Quick install (conda)

```bash
conda create -n streamlit-cars python=3.12
conda activate streamlit-cars
conda install -c conda-forge streamlit pandas numpy scikit-learn matplotlib pyarrow joblib
pip install category_encoders
```

### Or, with pip

```bash
pip install -r requirements.txt
```

## How to Run the App

1. Clone the repository and move into the folder:

   ```bash
   git clone <repo-URL>
   cd car_price_prediction
   ```

2. Check the file paths inside `streamlit_cars.py` (by default they point to absolute paths from the original machine, e.g. `/home/user/car_price_prediction/...`). Update them to relative paths or to your own local path, for example:

   ```python
   last_df = pd.read_parquet("datasets/pre_norm_dummied_df.parquet")
   loaded_model  = joblib.load("elastic_net_model.pkl")
   loaded_target = joblib.load("Target_encoder.pkl")
   loaded_scaler = joblib.load("StandardScaler.pkl")
   ```

3. Launch the app:

   ```bash
   streamlit run streamlit_cars.py
   ```

4. Open the URL shown in the terminal in your browser (by default `http://localhost:8501`)

## Retraining the Model

To retrain from scratch (new dataset, new hyperparameters, etc.), open and run `test_carDep.ipynb` from start to finish. On completion, the notebook regenerates and overwrites:

- `elastic_net_model.pkl`
- `StandardScaler.pkl`
- `Target_encoder.pkl`
- `sorted_coef_dict.pkl`

> **Important:** the order of the columns passed to the `StandardScaler` must always match the order it was trained on (`scaler.feature_names_in_`). The app already handles this automatically by reading that attribute, so no changes to `streamlit_cars.py` are needed after retraining.

## Known Limitations

- The model only covers the brands, colors, fuel types and transmissions present in the original dataset; values outside those categories are not supported by the form.
- The `.pkl` and `.parquet` file paths in `streamlit_cars.py` must be adapted to the deployment environment.
- The estimated price is indicative only, based purely on historical correlations in the Kaggle dataset.

## License

MIT License
