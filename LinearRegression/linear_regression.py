import pandas as pd
import plotly.express as px
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def simple_linear_regression(dataframe, dependent, independent, show_graph=False):
    """
    Simple linear regression is a form of linear regression where there is one dependent variable and one independent
    variable. It is represented by the equation of a line as follows:\n
    Y = mX + c\n
    X is the independent variable, Y is the dependent variable, m is the slope, c is the intercept.\n
    :param dataframe: Cleaned dataframe
    :param dependent: Dependent variable
    :param independent: Independent variable
    :param show_graph: Assign true to display graph
    :return: Slope, Intercept, Root Mean Squared Error, R2 Score
    """

    if show_graph:
        # Show the graph if specified
        fig = px.scatter(dataframe, x='Number of Claims', y='Total Payment', marginal_x='histogram', marginal_y='histogram')
        fig.show()

    # select the x and y values
    x = dataframe[dependent]
    y = dataframe[independent]

    # standardize the data attributes
    standardized_x = preprocessing.scale(x)

    # standardize the target attribute
    standardized_y = preprocessing.scale(y)

    # select training and testing data
    x_train, x_test, y_train, y_test = train_test_split(standardized_x, standardized_y, test_size=0.25, random_state=42)

    # reshape data into 2D array
    x_train = x_train.reshape(-1, 1)
    y_train = y_train.reshape(-1, 1)
    x_test = x_test.reshape(-1, 1)

    # Model initialization
    regression_model = LinearRegression()
    # Train the model
    regression_model.fit(x_train, y_train)
    # Predict
    y_predicted = regression_model.predict(x_test)

    # model evaluation
    slope = regression_model.coef_[0][0]
    intercept = regression_model.intercept_[0]
    root_mean_sq_err = mean_squared_error(y_test, y_predicted)
    r2 = r2_score(y_test, y_predicted)

    # print the values
    print('Slope:', slope)
    print('Intercept:', intercept)
    print('Root mean squared error: ', root_mean_sq_err)
    print('R2 score: ', r2)

    # return values
    return slope, intercept, root_mean_sq_err, r2


def run_simple_linear_regression():
    df = pd.read_csv("datasets/insurance.csv")
    simple_linear_regression(df, 'Number of Claims', 'Total Payment')


def multiple_linear_regression(dataframe, dependant):
    """
    Multiple linear regression is a form of linear regression where there are one dependent variable and several
    independent variables.\n
    It is represented by the equation of a line as follows:\n
    Y = m(X1+X2+X3...) + c\n
    X1, X2, X3 are the dependent variables, Y is the dependent variable, m1, m2, m3 are the coefficients of the
    independent variables that will be calculated.\n
    :param dataframe: Cleaned dataframe
    :param dependant: Dependent variable
    :return: Slope, Intercept, Root Mean Squared Error, R2 Score
    """

    if len(dataframe.drop(dependant, axis=1).columns) < 2:
        raise ValueError("Not enough independent variables, check how many columns in dataframe.")

    y = dataframe[dependant]
    x = dataframe.drop(dependant, axis=1)

    # standardize the data attributes
    standardized_x = preprocessing.scale(x)

    # standardize the target attribute
    standardized_y = preprocessing.scale(y)

    x_train, x_test, y_train, y_test = train_test_split(standardized_x, standardized_y, test_size=0.20, random_state=42)

    y_train = y_train.reshape(-1, 1)

    # Model initialization
    regression_model = LinearRegression()
    # Fit the data(train the model)
    regression_model.fit(x_train, y_train)
    # Predict
    y_predicted = regression_model.predict(x_test)

    # model evaluation
    slope = regression_model.coef_[0][0]
    intercept = regression_model.intercept_[0]
    root_mean_sq_err = mean_squared_error(y_test, y_predicted)
    r2 = r2_score(y_test, y_predicted)

    # print the values
    print('Slope:', slope)
    print('Intercept:', intercept)
    print('Root mean squared error: ', root_mean_sq_err)
    print('R2 score: ', r2)

    # return values
    return slope, intercept, root_mean_sq_err, r2


def run_multiple_linear_regression():
    data = pd.read_csv("datasets/car_price.csv")
    multiple_linear_regression(data, 'price')


if __name__ == "__main__":
    # simple
    # run_simple_linear_regression()
    # multiple
    run_multiple_linear_regression()
