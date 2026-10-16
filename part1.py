import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


# Mean Square Error
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Performance Evaluation
def r_squared(y_true, y_pred):
    return 1 - mse(y_true, y_pred) / np.var(y_true)


# Prediction
def pred(m, X, b):
    return X @ m + b

# gradient step
def gradient_step(X, y, w, b, learning_rate):
    n = len(y)

    # Predictions
    y_pred = pred(w, X, b)

    # Errors
    error = y_pred - y

    # Gradients
    dw = (2 / n) * (X.T @ error)
    db = (2 / n) * np.sum(error)

    # Update parameters
    w = w - learning_rate * dw
    b = b - learning_rate * db

    return w, b



def gradient_descent(X, y, learning_rate, iterations):
    w = np.zeros(X.shape[1])
    b = 0.0
    loss_history = []

    for i in range(iterations):
        w, b = gradient_step(X, y, w, b, learning_rate)

        # Record training MSE after each update
        loss = mse(y, pred(w, X, b))
        loss_history.append(loss)

    return w, b, loss_history



if __name__ == "__main__":
    lr_raw = 0.0000001
    lr_scaled = 0.001
    lr_log = 0.001
    iterations = 10000

    df = pd.read_excel("/Users/xinshifeng/Downloads/concrete+compressive+strength/Concrete_Data.xls")

    # Split into X (input) and y (output)
    X = df.iloc[:, :-1].to_numpy()
    y = df.iloc[:, -1].to_numpy()

    # Testing set: rows 501–630
    X_test = X[500:630]
    y_test = y[500:630]

    # Training set: all remaining rows
    X_train = np.concatenate((X[:500], X[630:]), axis=0)
    y_train = np.concatenate((y[:500], y[630:]), axis=0)

    # Calculate mean and standard deviation for each feature
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    # Standardize using training statistics
    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std

    # Logarithmic transformation
    X_train_log = np.log1p(X_train)
    X_test_log = np.log1p(X_test)





    # Q3
    print("\n----------Original Model----------")
    w_raw, b_raw, loss_raw = gradient_descent(
        X_train,
        y_train,
        learning_rate=lr_raw,
        iterations=iterations
    )

    y_raw_pred = pred(w_raw, X_train, b_raw)

    print("\nUntransformed Model")
    print("Training MSE:", mse(y_train, y_raw_pred))
    print("Training R²:", r_squared(y_train, y_raw_pred))


    w_scaled, b_scaled, loss_scaled = gradient_descent(
        X_train_scaled,
        y_train,
        learning_rate=lr_scaled,
        iterations=iterations
    )

    y_scaled_pred = pred(w_scaled, X_train_scaled, b_scaled)

    print("\nStandardized Model")
    print("Training MSE:", mse(y_train, y_scaled_pred))
    print("Training R²:", r_squared(y_train, y_scaled_pred))

    plt.figure(figsize=(10, 6))

    plt.plot(loss_raw, label=f"Untransformed features (lr={lr_raw})")
    plt.plot(loss_scaled, label=f"Standardized features (lr={lr_scaled})")

    plt.xlabel("Iteration")
    plt.ylabel("Training MSE (log scale)")
    plt.yscale("log")

    plt.title("Gradient Descent: Scaling vs No Scaling")
    plt.legend()
    plt.grid(True)

    plt.show()


    # Q4
    print("\n----------External Model----------")
    # Untransformed model
    model_raw = LinearRegression()
    model_raw.fit(X_train, y_train)

    y_pred_raw = model_raw.predict(X_train)

    print("\nUntransformed Model")
    print("Training MSE:", mse(y_train, y_pred_raw))
    print("Training R²:", r_squared(y_train, y_pred_raw))

    # Standardized model
    model_scaled = LinearRegression()
    model_scaled.fit(X_train_scaled, y_train)

    y_pred_scaled = model_scaled.predict(X_train_scaled)

    print("\nStandardized Model")
    print("Training MSE:", mse(y_train, y_pred_scaled))
    print("Training R²:", r_squared(y_train, y_pred_scaled))



    # Q5
    print("\n----------Log Transformed----------")
    w_log, b_log, loss_log = gradient_descent(
        X_train_log,
        y_train,
        learning_rate=lr_log,
        iterations=iterations
    )

    # Evaluate
    y_train_log_pred = pred(w_log, X_train_log, b_log)
    y_test_log_pred = pred(w_log, X_test_log, b_log)


    print("Logarithmic Model")
    print("Training MSE:", mse(y_train, y_train_log_pred))
    print("Training R²:", r_squared(y_train, y_train_log_pred))
    print("Testing MSE:", mse(y_test, y_test_log_pred))
    print("Testing R²:", r_squared(y_test, y_test_log_pred))


    plt.figure(figsize=(10, 6))

    plt.plot(loss_raw, label=f"Untransformed (lr={lr_raw})")
    plt.plot(loss_scaled, label=f"Standardized (lr={lr_scaled})")
    plt.plot(loss_log, label=f"Log (lr={lr_log})")

    plt.xlabel("Iteration")
    plt.ylabel("Training MSE")
    plt.yscale("log")
    plt.title("Gradient Descent Convergence Comparison")
    plt.legend()
    plt.grid(True)

    plt.show()


    # 1. Untransformed model
    model_raw = LinearRegression()
    model_raw.fit(X_train, y_train)

    # 2. Standardized model
    model_scaled = LinearRegression()
    model_scaled.fit(X_train_scaled, y_train)

    # 3. Log-transformed model
    X_train_log = np.log1p(X_train)
    X_test_log = np.log1p(X_test)

    model_log = LinearRegression()
    model_log.fit(X_train_log, y_train)


    # Evaluate a model
    def evaluate_model(model, X_train, X_test, y_train, y_test):
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        return {
            "Training MSE": mse(y_train, y_train_pred),
            "Testing MSE": mse(y_test, y_test_pred),
            "Training R²": r_squared(y_train, y_train_pred),
            "Testing R²": r_squared(y_test, y_test_pred)
        }


    # Summary table
    results = pd.DataFrame({
        "Untransformed": evaluate_model(
            model_raw, X_train, X_test, y_train, y_test
        ),
        "Standardized": evaluate_model(
            model_scaled, X_train_scaled, X_test_scaled,
            y_train, y_test
        ),
        "Log-Transformed": evaluate_model(
            model_log, X_train_log, X_test_log,
            y_train, y_test
        )
    }).T

    print()
    print(results.round(4))

    # Q6
    print("\n----------Univariate Models----------")

    feature_names = df.columns[:-1]

    univariate_results = []

    for i in range(X_train.shape[1]):
        # Select one feature, preserving a 2D array
        X_train_uni = X_train[:, i:i + 1]
        X_test_uni = X_test[:, i:i + 1]

        # learning rate based on the feature magnitude
        lr = 0.1 / np.mean(X_train_uni[:, 0] ** 2)

        # Train using our gradient descent implementation
        w, b, loss_history = gradient_descent(
            X_train_uni,
            y_train,
            learning_rate=lr,
            iterations=100000
        )

        # Predictions
        y_train_pred = pred(w, X_train_uni, b)
        y_test_pred = pred(w, X_test_uni, b)

        # Store results
        univariate_results.append({
            "Feature": feature_names[i],
            "Learning Rate": lr,
            "Coefficient": w[0],
            "Intercept": b,
            "Training MSE": mse(y_train, y_train_pred),
            "Training R²": r_squared(y_train, y_train_pred),
            "Testing R²": r_squared(y_test, y_test_pred)
        })

    # Display results
    uni_results = pd.DataFrame(univariate_results)

    print(uni_results.round(5))


    print("\n----------Univariate vs Multivariate----------")

    feature_names = df.columns[:-1]

    # Optimal multivariate model (untransformed)
    model_multi = LinearRegression()
    model_multi.fit(X_train, y_train)

    multi_coefficients = model_multi.coef_

    comparison = []

    for i in range(X_train.shape[1]):
        # Select only one predictor
        X_uni = X_train[:, i:i + 1]

        # Fit optimal univariate model
        model_uni = LinearRegression()
        model_uni.fit(X_uni, y_train)

        comparison.append({
            "Feature": feature_names[i],
            "Univariate Coefficient": model_uni.coef_[0],
            "Multivariate Coefficient": multi_coefficients[i],
            "Difference": model_uni.coef_[0] - multi_coefficients[i]
        })

    # Create comparison table
    comparison_df = pd.DataFrame(comparison)

    print(comparison_df.to_string(index=False))


    # Q7
    def get_pvalues(X_data, y_data):
        # Add the intercept required by OLS
        X_with_intercept = sm.add_constant(
            X_data, has_constant="add"
        )

        model = sm.OLS(y_data, X_with_intercept).fit()

        # Exclude the intercept p-value
        return model.pvalues[1:]


    short_feature_names = [
        "Cement",
        "Blast Furnace Slag",
        "Fly Ash",
        "Water",
        "Superplasticizer",
        "Coarse Aggregate",
        "Fine Aggregate",
        "Age"
    ]

    pvalue_results = pd.DataFrame(
        {
            "Untransformed": get_pvalues(X_train, y_train),
            "Standardized": get_pvalues(X_train_scaled, y_train),
            "Log-Transformed": get_pvalues(X_train_log, y_train)
        },
        index=short_feature_names
    )

    print("\n----------Q7: Feature p-values----------")
    print(pvalue_results)