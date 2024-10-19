
# Wind Power Forecasting using Machine Learning Models

This project implements a machine learning framework to forecast wind power generation using various machine learning models. The dataset used is **wtk_site_metadata.csv**, containing wind power and related features. This project compares the performance of several models, including:

- **Linear Regression**
- **Random Forest**
- **Gradient Boosting**
- **XGBoost**
- **LightGBM**

The models are evaluated using key metrics: **Root Mean Squared Error (RMSE)**, **R-squared**, and **Mean Absolute Error (MAE)**.

## Installation

To run this project, ensure you have the following packages installed:

```bash
pip install pandas numpy scikit-learn matplotlib xgboost lightgbm
```

## Usage

1. **Load Dataset**: 
   The dataset `wtk_site_metadata.csv` is loaded and preprocessed. Features include wind speed, capacity, and geographical information.

2. **Feature Engineering**:
   Latitude and longitude are multiplied by wind speed to create two new features: `latitude_wind` and `longitude_wind`.

3. **Data Cleaning**:
   Missing values are dropped from key columns.

4. **Train-Test Split**:
   The dataset is split into training (70%) and testing (30%) sets using `train_test_split`.

5. **Model Training**:
   Five machine learning models are trained:
   - Linear Regression
   - RandomForestRegressor
   - GradientBoostingRegressor
   - XGBRegressor (XGBoost)
   - LGBMRegressor (LightGBM)

6. **Model Evaluation**:
   Each model is evaluated based on RMSE, R-squared, and MAE, and the results are stored for comparison.

7. **Visualization**:
   The project generates several bar plots to compare model performance:
   - RMSE Comparison
   - R-squared Comparison
   - MAE Comparison
   - Performance improvement of advanced models over Linear Regression (in RMSE)
   - ROC-like curve showing error rate vs. threshold for each model.

## Results

The results are printed and visualized using Matplotlib. The final output includes:

- A comparison of **RMSE**, **R-squared**, and **MAE** across all models.
- **Performance Improvement** over Linear Regression is calculated and plotted.
- A **ROC-like curve** is plotted to evaluate the error rates for various thresholds.

## Example Outputs

1. **RMSE Comparison**:
   This plot shows the RMSE of each model. Lower RMSE indicates better performance.

2. **R-squared Comparison**:
   Higher R-squared values indicate a better fit for the data.

3. **MAE Comparison**:
   Lower MAE values indicate more accurate predictions.

4. **Performance Improvement**:
   Bar plot comparing the percentage improvement in RMSE over Linear Regression.

5. **ROC-like Curve**:
   Error rate vs. threshold curves for each model, showcasing model robustness.

## How to Run

1. Place your dataset in the working directory with the filename `wtk_site_metadata.csv`.
2. Run the Python script:

   ```bash
   python wind_power_forecasting.py
   ```

3. The results will be printed to the console and visualized using Matplotlib plots.

## Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **scikit-learn**: Machine learning algorithms and evaluation metrics
- **xgboost**: XGBoost implementation for gradient boosting
- **lightgbm**: LightGBM for fast gradient boosting
- **matplotlib**: Data visualization

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

Special thanks to the data providers and the open-source community for the machine learning libraries.
