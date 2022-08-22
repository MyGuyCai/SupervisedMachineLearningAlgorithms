import pandas as pd
from math import ceil
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder


def logistic_regression(data_frame, dependent: str, columns_to_inspect: list = None, show_correlation: bool = False):
    """
    Logistic regression equations are an improvised version of linear regression but far more flexible due to the\n
    seperation of the graph.\n
    Our logistic regression equation can also be represented as follows:\n
    z = log(y/(1-y))\n
    In this case, z is also known as log-odds, as it gives us the log of the odds of an event happening.\n
    The expression ( y/ (1-y)) is known as the odds. In logistic regression, the dependent variable is logit, which is the\n
    natural log(ln) of the odds\n
    :param data_frame: Cleaned dataframe
    :param dependent: Dependent variable
    :param columns_to_inspect: Pass a list to view graphs
    :param show_correlation: Set true to view correlation
    :return: Accuracy and confusion matrix
    """
    def display_graphs(col_to_d, dep):
        def calculate_rows_columns(col_names, col):
            # calculate the total number of unique graphs possible
            total_graphs = 0
            for cl in col_names[1:-1]:
                total_graphs += len(col_names) - col_names.index(cl)
            # here we add one as we will be showing a graph of the dependent data
            total_graphs += 1
            row_c = ceil(total_graphs / col) + 1

            spec_c = []
            for s in range(columns):
                if s == 0:
                    spec_c.append([{'colspan': columns}])
                else:
                    spec_c[0].append(None)
            for c in range(row_c - 1):
                sp = []
                for r in range(columns):
                    sp.append({})
                spec_c.append(sp)

            return row_c, spec_c

        data_frame_n = data_frame[col_to_d]

        columns = 5
        rows, spec = calculate_rows_columns(col_to_d, columns)
        fig = make_subplots(rows=rows, cols=columns, specs=spec)

        fig.add_trace(
            go.Histogram(x=data_frame[dep], name=dep),
            row=1, col=1)

        row, column = 2, 1
        for index_x, value_x in enumerate(col_to_d):
            used_x = col_to_d[index_x]
            for index_y, value_y in enumerate(col_to_d[:-index_x + 1]):
                fig.add_trace(
                    go.Scatter(x=data_frame_n[used_x], y=data_frame_n[value_y], mode="markers",
                               name=used_x + " " + value_y),
                    row=row, col=column)
                column += 1
                if column > columns:
                    column = 1
                    row += 1

        fig.show()

    # First we clean the data
    data_frame = data_frame.drop(columns=['Unnamed: 32'])

    # Now we need to visualise our data to check for possible relationships between data
    if columns_to_inspect:
        display_graphs(columns_to_inspect, dependent)

    if show_correlation:
        # view correlation
        corr = data_frame.corr()

        corr_fig = px.imshow(corr)
        corr_fig.show()

    # the correlation matrix shows that we can drop many columns as they show almost the same thing
    cols_drop = ['perimeter_mean', 'area_mean', 'compactness_mean', 'concave points_mean', 'radius_se', 'perimeter_se',
                 'radius_worst', 'perimeter_worst', 'compactness_worst', 'concave points_worst', 'compactness_se',
                 'concave points_se', 'texture_worst', 'area_worst']
    data_frame = data_frame.drop(cols_drop, axis=1)

    # Next we need to encode the diagnosis data to values
    le = LabelEncoder()
    data_frame[dependent] = le.fit_transform(data_frame[dependent])

    y = data_frame[dependent]
    x = data_frame.drop(columns=[dependent])

    # Now we can select the data for training and testing
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

    # create an instance and fit the model
    logit = LogisticRegression()
    logit.fit(x_train, y_train)

    y_predict = logit.predict(x_test)

    # Now we can show the accuracy of the prediction model
    report = classification_report(y_test, y_predict)

    # And finally view the False positive values
    confusion_mat = confusion_matrix(y_test, y_predict)
    confusion_dict = {
        'true_positives': confusion_mat[0][0],
        'false_positives': confusion_mat[0][1],
        'true_negatives': confusion_mat[1][0],
        'false_negatives': confusion_mat[1][1],
    }

    return report, confusion_dict


if __name__ == "__main__":
    df = pd.read_csv('datasets/data.csv', index_col='id')
    cols = ['radius_mean', 'smoothness_mean', 'compactness_mean', 'fractal_dimension_mean', 'area_mean']
    accuracy, confusion = logistic_regression(df, 'diagnosis', cols, True)
    print(accuracy)
    print(confusion)
