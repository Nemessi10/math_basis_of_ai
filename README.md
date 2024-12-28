# Documentation for Restaurant Revenue Prediction Model

## Project Overview
This project focuses on building a machine learning model for predicting restaurant revenues based on various features, including city, type, and opening date. The data comes from a restaurant revenue prediction challenge, and the project applies several regression techniques to predict the revenue and evaluate the model's performance.

## Project Requirements
To run the project, you need to install the required libraries using the following commands:

```bash
!pip install xgboost lightgbm catboost
!pip install category_encoders
```

Libraries used in this project:
- **XGBoost, LightGBM, CatBoost**: Gradient boosting algorithms for regression tasks.
- **category_encoders**: Encoding categorical variables for machine learning models.
- **scikit-learn**: For model evaluation, splitting the dataset, and preprocessing data.
- **matplotlib, seaborn**: For plotting and visualizing results.

## File Structure
- **train.csv**: Training dataset containing information about restaurant locations and revenues.
- **test.csv**: Testing dataset, without revenue information, used for predictions.
- **RestaurantRevenuePrediction**: Folder containing the datasets and the code for training and evaluating models.

## Main Components

### 1. Data Preprocessing
The preprocessing pipeline includes:
- **Adding date-related features**: The `add_datepart` function extracts additional features from the `Open Date` column, such as year, month, day, and whether the date is at the start or end of a month or year.
- **Handling missing values**: Missing values are imputed depending on the data type. Categorical columns are filled with 'missing', and numerical columns are filled with the median value.
- **Categorical encoding**: The `convert_cats` function encodes categorical columns using integer encoding for object-type columns and binary encoding for the 'City' column.
- **Feature selection**: Certain columns (e.g., 'Id', 'Open Date') are dropped to avoid overfitting and simplify the model.

### 2. Data Splitting and Feature Engineering
The function `merge_and_process_data` merges the training and test data and applies feature engineering:
- **Encoding categorical columns** and handling missing values.
- **Feature elimination**: Columns such as 'City', 'Open Date', and others are removed as they are not useful for model training.
- **Scaling**: The features are scaled using `StandardScaler` to improve model performance.

### 3. Model Training
Various machine learning models are trained and evaluated:
- **Random Forest**, **XGBoost**, **LightGBM**, **CatBoost**: Popular gradient boosting and ensemble models for regression tasks.
- **Cross-validation**: The models are trained using 4-fold cross-validation. The `train_cv` function evaluates the models using RMSE (Root Mean Squared Error).
- **Neural Network**: A custom neural network class is implemented using a basic feedforward structure with ReLU activation function.

### 4. Model Evaluation
The evaluation process involves:
- **Root Mean Squared Error (RMSE)**: Used to assess the accuracy of predictions. The `calculate_rmse` function computes RMSE between the actual and predicted values.
- **Visualization**: A bar plot is generated to compare the models' performance based on their RMSE values (`plot_model_performance` function).
  
### 5. Prediction
After training the models, the final predictions are made using a custom neural network (`CustomNeuralNetwork`) and ensemble methods. The predictions are scaled back to the original scale using the exponential function.

### 6. Custom Neural Network
The `CustomNeuralNetwork` class implements a simple neural network with the following features:
- **ReLU activation function**: For forward propagation.
- **Gradient descent**: Used for optimizing the weights during the backward pass.
- **Training and prediction**: The network is trained using the `fit` method and makes predictions using the `predict` method.

## Functions

### `score(model, X_train, y_train, X_valid = [], y_valid = [])`
Evaluates the model's performance on both training and validation data using RMSE.

### `add_datepart(df, date_col)`
Adds features derived from the date column (`date_col`) such as year, month, day, etc.

### `convert_cats(df)`
Converts categorical columns to integer categories and binary encodes the 'City' column.

### `clean_and_split_df(df, y_col=None)`
Handles missing values, drops unnecessary columns, and splits the dataframe into features (`X`) and target (`y`).

### `merge_and_process_data(df_train, df_test, fe=[])`
Merges and preprocesses the training and test data, applying feature engineering and scaling.

### `train_cv(X, y, model_list)`
Trains multiple models using cross-validation and returns the models' performance based on the RMSE metric.

### `plot_model_performance(results)`
Generates a bar plot comparing model performances based on RMSE.

### `predict(models, X)`
Makes predictions using an ensemble of trained models.

### `calculate_rmse(y_true, y_pred)`
Calculates RMSE between the true and predicted values.

### `CustomNeuralNetwork` Class
A custom neural network implementation with the following methods:
- **`__init__`**: Initializes the network with random weights and biases.
- ** `relu`**: Activation function.
- **`forward`**: Forward pass through the network.
- **`backward`**: Backward pass using gradient descent.
- **`fit`**: Trains the network.
- **`predict`**: Makes predictions using the trained model.

## Running the Model
1. Import the data:
   - `df_train` and `df_test` are loaded from CSV files.
2. Preprocess the data:
   - The `merge_and_process_data` function combines and preprocesses the training and test data.
3. Train the models:
   - Multiple models (Random Forest, XGBoost, LightGBM, CatBoost, and a custom neural network) are trained using cross-validation.
4. Evaluate the models:
   - The `plot_model_performance` function displays the comparison of models based on RMSE.
5. Make predictions:
   - The trained models are used to predict the revenue for the test dataset.

## Conclusion
This project demonstrates the use of various machine learning models for predicting restaurant revenues, with a focus on gradient boosting methods and neural networks. By following this documentation, even someone unfamiliar with the code can set up, train, and evaluate the models, as well as make predictions on new data.
