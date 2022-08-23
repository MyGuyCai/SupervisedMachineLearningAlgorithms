import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import svm


class SupportVectorMachine:
    def __init__(self, csv_path, dependent, index_col=None):
        self.data_frame = pd.read_csv(csv_path, index_col=index_col)
        self.dependent = dependent

    def show_graphs(self, x):
        fig = px.box(self.data_frame, x, self.dependent)
        fig.show()

    def run_modelling(self):
        y = self.data_frame[self.dependent]
        x = self.data_frame.drop(columns=[self.dependent])

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
        model_svm = svm.SVC(kernel='linear')

        model_svm.fit(x_train, y_train)

        y_predicted = model_svm.predict(x_test)

        print(classification_report(y_test, y_predicted))

        print(confusion_matrix(y_test, y_predicted))


if __name__ == "__main__":
    vector_machine = SupportVectorMachine('datasets/Iris.csv', 'Species', 'Id')
    # vector_machine.show_graphs('SepalLengthCm')
    # vector_machine.show_graphs('SepalWidthCm')
    vector_machine.run_modelling()
