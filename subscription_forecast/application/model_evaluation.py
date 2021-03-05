
from sklearn.metrics import recall_score,\
    precision_score, \
    confusion_matrix, \
    accuracy_score, \
    precision_recall_curve
import matplotlib.pyplot as plt


class ModelEvaluator:
    """Class called for a pipeline and and a type of model.
    Provides metrics to evaluate and visualize """
    def __init__(self, model_type: str, pipeline: object):
        self.model_type = model_type
        self.pipeline = pipeline

    def print_metrics(self, x_test, y_test):
        """Display various metrics to evaluate the performance of the model:
            -Accuracy
            -Precision
            -Recall
            -Confusion Matrix"""
        y_test = y_test.to_numpy()
        y_pred = self.pipeline.predict(x_test)

        print("Model accuracy : ", accuracy_score(y_test, y_pred))
        print("Model precision : ", precision_score(y_test, y_pred, average="binary", pos_label='Yes'))
        print("Model recall = ", recall_score(y_test, y_pred, average="binary", pos_label='Yes'))
        print(confusion_matrix(y_test, y_pred, labels=['Yes', 'No']))

        # TODO: put this in an interpretability object/function
        print("Feature importance: ", self.pipeline.steps[1][1].feature_importances_)

    def plot_precision_recall(self, x_test, y_test, ColumnTransformer):
        """Get the engineered features and plot the precision recall curve for the model"""
        # get features
        transformer = ColumnTransformer
        X_FE = transformer.fit_transform(x_test)
        # get precision and recall
        y_score = self.pipeline.steps[1][1].predict_proba(X_FE)
        precision, recall, thresholds = precision_recall_curve(y_test, y_score[:, 1], pos_label='Yes')
        # plot results
        plt.plot(recall, precision)
        plt.title(f'Precision recall curve for the {self.model_type} classifier')
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.show()
