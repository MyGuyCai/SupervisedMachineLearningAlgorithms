import pandas as pd
import plotly.express as px
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


class RandomForestModel:
    def __init__(self, csv_path, dependent):
        self.data_frame = pd.read_csv(csv_path)
        self.dependent = dependent

    def change_headers(self, headers):
        self.data_frame.columns = headers

    def display_correlation(self):
        corr = self.data_frame.corr()

        corr_fig = px.imshow(corr)
        corr_fig.show()

    def drop_useless(self, columns):
        self.data_frame = self.data_frame.drop(columns=columns)
        self.data_frame.fillna(method='ffill', inplace=True)

    def run_modelling(self):
        y = self.data_frame[self.dependent]
        x = self.data_frame.drop(columns=[self.dependent])

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

        # create the random forest classifier
        classifier = RandomForestClassifier(n_jobs=2, random_state=42)
        # fit to existing training set
        classifier.fit(x_train, y_train)
        # Predict the response for test dataset
        y_pred = classifier.predict(x_test)
        # Model Accuracy
        print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

        confusion_matrix_dict = confusion_matrix(y_test, y_pred, labels=None, sample_weight=None)
        print(confusion_matrix_dict)


if __name__ == "__main__":
    model = RandomForestModel('datasets/Wine.csv', 'Class')
    attributes = ["Class", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols",
                  "Flavanoids", "Nonflavanoidphenols", "Proanthocyanins", "Color intensity", "Hue",
                  "OD280OD315 of diluted wines", "Proline"]
    model.change_headers(attributes)
    model.display_correlation()
    model.drop_useless('Flavanoids')
    model.run_modelling()
