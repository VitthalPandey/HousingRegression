# regression.py
from utils import load_data
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore")


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name}: MSE = {mse:.2f}, R² = {r2:.2f}")


def hyperparameter_tuning(model, params, X_train, y_train):
    grid = GridSearchCV(model, params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(X_train, y_train)
    print(f"Best parameters for {model.__class__.__name__}: {grid.best_params_}")
    return grid.best_estimator_


def main():
    df = load_data()
    X = df.drop('MEDV', axis=1)
    y = df['MEDV']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Linear Regression (limited hyperparameters)
    lr = LinearRegression()
    lr_params = {
        'fit_intercept': [True, False],
        'positive': [True, False],
        'copy_X': [True, False]
    }

    # Decision Tree
    dt = DecisionTreeRegressor(random_state=42)
    dt_params = {
        'max_depth': [3, 5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Random Forest
    rf = RandomForestRegressor(random_state=42)
    rf_params = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10, 20],
        'min_samples_split': [2, 5]
    }

    # Tune and evaluate all models
    best_lr = hyperparameter_tuning(lr, lr_params, X_train, y_train)
    evaluate_model("Linear Regression (Tuned)", best_lr, X_test, y_test)

    best_dt = hyperparameter_tuning(dt, dt_params, X_train, y_train)
    evaluate_model("Decision Tree (Tuned)", best_dt, X_test, y_test)

    best_rf = hyperparameter_tuning(rf, rf_params, X_train, y_train)
    evaluate_model("Random Forest (Tuned)", best_rf, X_test, y_test)


if __name__ == "__main__":
    main()
