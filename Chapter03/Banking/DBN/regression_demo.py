import numpy as np

np.random.seed(1337)  # for reproducibility
from sklearn.datasets import load_boston # Load the boston dataset from sklearn
from sklearn.model_selection import train_test_split

#Import R2 and MSE metrics from sklearn
from sklearn.metrics.regression import r2_score, mean_squared_error 
from sklearn.preprocessing import MinMaxScaler # For data scaling 

# For regression
from tf_models import SupervisedDBNRegression


# Loading dataset
boston = load_boston()
X, Y = boston.data, boston.target

# Splitting data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1337)

# Data scaling
min_max_scaler = MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)

# Training
regressor = SupervisedDBNRegression(hidden_layers_structure=[100, 100],
                                    learning_rate_rbm=0.10,
                                    learning_rate=0.01,
                                    n_epochs_rbm=10,
                                    n_iter_backprop=10,
                                    batch_size=16,
                                    activation_function='relu')
regressor.fit(X_train, Y_train)

# Test
X_test = min_max_scaler.transform(X_test)
Y_pred = regressor.predict(X_test)
print('Done.\nR-squared: %f\nMSE: %f' % (r2_score(Y_test, Y_pred), mean_squared_error(Y_test, Y_pred)))
