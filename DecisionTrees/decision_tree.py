import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
import plotly.express as px
from pandas.api.types import is_int64_dtype, is_float_dtype


class DecisionTree:
    def __init__(self, path, dependent):
        self.data_frame = pd.read_csv(path)
        self.dependent = dependent

    def drop_useless(self, columns):
        self.data_frame = self.data_frame.drop(columns=columns)
        self.data_frame.fillna(method='ffill', inplace=True)

    def encode_objects(self):
        for col in self.data_frame.columns:
            if not is_int64_dtype(col) or not is_float_dtype(col):
                lab_enc = preprocessing.LabelEncoder()
                self.data_frame[col] = lab_enc.fit_transform(self.data_frame[col])
        print(self.data_frame)

    def display_correlation(self):
        corr = self.data_frame.corr()

        corr_fig = px.imshow(corr)
        corr_fig.show()

    def run_modelling(self):
        y = self.data_frame[self.dependent]
        x = self.data_frame.drop(columns=[self.dependent])

        # split the data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 42)

        # Create Decision Tree classifier object
        clf = DecisionTreeClassifier()

        # Train Decision Tree Classifier
        clf = clf.fit(x_train, y_train)

        # Predict the response for test dataset
        y_prediction = clf.predict(x_test)

        # Model Accuracy, how often is the classifier correct?
        print("Accuracy:", metrics.accuracy_score(y_test, y_prediction))


if __name__ == '__main__':
    model = DecisionTree('datasets/titanic.csv', 'Survived')
    drop_cols = ['Name', 'Siblings/Spouses Aboard', 'Parents/Children Aboard']
    model.drop_useless(drop_cols)
    model.encode_objects()
    model.run_modelling()



