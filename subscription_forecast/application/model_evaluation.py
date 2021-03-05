
from sklearn.metrics import recall_score,\
    precision_score, \
    confusion_matrix, \
    accuracy_score, \
    precision_recall_curve
import matplotlib.pyplot as plt


class ModelEvaluator:
    """ class called for a pipeline and and a type of model.
    Provides metrics to evaluate and visualize """
    def __init__(self, model: str, pipeline: object):
        self.model = model
        self.pipeline = pipeline

    def print_score(self, X_test, y_test):
        print(self.pipeline.score(X_test, y_test))

        y_test = y_test.to_numpy()
        y_pred = self.pipeline.predict(X_test)

        print("Model accuracy : ", accuracy_score(y_test, y_pred))
        print("Model arecision : ", precision_score(y_test, y_pred, average="binary", pos_label='Yes'))
        print("Model recall = ", recall_score(y_test, y_pred, average="binary", pos_label='Yes'))
        print("Feature importances: ", self.pipeline.steps[1][1].feature_importances_)
        print(confusion_matrix(y_test, y_pred, labels=['Yes', 'No']))


    def plot_precision_recall(self, transformer: object):

        X_FE = transformer.fit_transform(self.X_test)

        y_score = self.pipeline.steps[1][1].predict_proba(X_FE)
        precision, recall, thresholds = precision_recall_curve(self.y_test, y_score[:, 1], pos_label='Yes')

        plt.plot(recall, precision)
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.show()
