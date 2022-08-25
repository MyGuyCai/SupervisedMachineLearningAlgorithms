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


class LogisticRegressionModel:
    test_size = 0.33
    random_state = 42

    def __init__(self, csv_path: str, dependent: str, index_col: str = None, drop_cols: list = None):
        """
        Logistic regression equations are an improvised version of linear regression but far more flexible due to
        the separation of the graph.\n
        Our logistic regression equation can also be represented as follows:\n
        z = log(y/(1-y))\n
        In this case, z is also known as log-odds, as it gives us the log of the odds of an event happening.\n
        The expression ( y/ (1-y)) is known as the odds. In logistic regression, the dependent variable is logit, \n
        which is the natural log(ln) of the odds\n
        :param csv_path: Path to csv file
        :param dependent: Column name for dependent
        :param index_col: Specify if data has index column
        :param drop_cols: Columns that should be dropped for cleaning
        """
        self.confusion_dict = None
        self.report = None
        self.data_frame = pd.read_csv(csv_path, index_col=index_col)
        self.dependent = dependent

        if drop_cols:
            self.drop_columns(drop_cols)

        self._check_data()

    def drop_columns(self, drop_cols: list):
        """
        Drop columns by name from the dataframe\n
        :param drop_cols: List of columns to drop
        """
        self.data_frame = self.data_frame.drop(columns=drop_cols)

    def _check_data(self):
        column_count = set()
        for col in self.data_frame.columns:
            column_count.add(self.data_frame[col].count())
        if len(column_count) > 1:
            raise TypeError("Dataframe needs to be cleaned")

    def display_graphs(self, column_names):
        """
        Show all combinations of graph for a given list of column names\n
        :param column_names: List of column names
        :return: Display graph in browser
        """
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
            for _ in range(row_c - 1):
                sp = []
                for _ in range(columns):
                    sp.append({})
                spec_c.append(sp)

            return row_c, spec_c

        data_frame_n = self.data_frame[column_names]

        columns = 5
        rows, spec = calculate_rows_columns(column_names, columns)
        fig = make_subplots(rows=rows, cols=columns, specs=spec)

        fig.add_trace(
            go.Histogram(x=self.data_frame[self.dependent], name=self.dependent),
            row=1, col=1)

        row, column = 2, 1
        for index_x, value_x in enumerate(column_names):
            used_x = column_names[index_x]
            for index_y, value_y in enumerate(column_names[:-index_x + 1]):
                fig.add_trace(
                    go.Scatter(x=data_frame_n[used_x], y=data_frame_n[value_y], mode="markers",
                               name=used_x + " " + value_y),
                    row=row, col=column)
                column += 1
                if column > columns:
                    column = 1
                    row += 1

        fig.show()

    def display_correlation(self):
        """
        Display a heatmap for all values in the dataframe, to exclude similar data from processing.
        :return: Display graph in browser.
        """
        corr = self.data_frame.corr()

        corr_fig = px.imshow(corr)
        corr_fig.show()

    def run_modelling(self):
        """
        Run the regression Model.
        """
        # Next we need to encode the diagnosis data to values
        le = LabelEncoder()
        self.data_frame[self.dependent] = le.fit_transform(self.data_frame[self.dependent])

        y = self.data_frame[self.dependent]
        x = self.data_frame.drop(columns=[self.dependent])

        # Now we can select the data for training and testing
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=LogisticRegressionModel.test_size,
                                                            random_state=LogisticRegressionModel.random_state)

        # create an instance and fit the model
        logit = LogisticRegression()
        logit.fit(x_train, y_train)

        y_predict = logit.predict(x_test)

        # Now we can show the accuracy of the prediction model
        self.report = classification_report(y_test, y_predict)

        # And finally view the False positive values
        confusion_mat = confusion_matrix(y_test, y_predict)
        self.confusion_dict = {
            'true_positives': confusion_mat[0][0],
            'false_positives': confusion_mat[0][1],
            'true_negatives': confusion_mat[1][0],
            'false_negatives': confusion_mat[1][1],
        }


if __name__ == "__main__":
    # first we set up the object by passing the data
    model = LogisticRegressionModel(csv_path='datasets/data.csv', dependent='diagnosis', index_col='id',
                                    drop_cols=['Unnamed: 32'])

    # next we analyse the data to see trends
    cols = ['radius_mean', 'smoothness_mean', 'compactness_mean', 'fractal_dimension_mean', 'area_mean']
    model.display_graphs(cols)

    # now we need to clean up the data by analysing less useful stats
    model.display_correlation()
    # the correlation matrix shows that we can drop many columns as they show almost the same thing
    cols_drop = ['perimeter_mean', 'area_mean', 'compactness_mean', 'concave points_mean', 'radius_se', 'perimeter_se',
                 'radius_worst', 'perimeter_worst', 'compactness_worst', 'concave points_worst', 'compactness_se',
                 'concave points_se', 'texture_worst', 'area_worst']
    model.drop_columns(cols_drop)

    # finally we can run the modelling
    model.run_modelling()

    # printing the results of our LogisticRegression
    print(model.report)
    print(model.confusion_dict)
