def hyperparameter_tuning():
    x_train, y_train, _, _ = gen_data(750)
    X_train, X_val_test, y_train, y_val_test = train_test_split(x_train, y_train, test_size=0.4, random_state=1)
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=1)

    degrees = range(1, 16)
    lambdas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 300, 600, 900]
    best_degree = None
    best_lambda = None
    min_test_error = float('inf')

    for degree in degrees:
        for lmbda in lambdas:
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            X_train_poly = poly.fit_transform(X_train[:, np.newaxis])
            X_val_poly = poly.transform(X_val[:, np.newaxis])

            model = Ridge(alpha=lmbda)
            model.fit(X_train_poly, y_train)

            y_val_pred = model.predict(X_val_poly)
            test_error = mean_squared_error(y_val, y_val_pred)

            if test_error < min_test_error:
                min_test_error = test_error
                best_degree = degree
                best_lambda = lmbda

    print("Best polynomial degree:", best_degree)
    print("Best lambda:", best_lambda)