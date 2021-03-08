
from sklearn.metrics import recall_score,\
    precision_score, \
    confusion_matrix, \
    accuracy_score, \
    precision_recall_curve, \
    auc
import matplotlib.pyplot as plt
import numpy as np


class ModelEvaluator:
    """Class called for a pipeline and and a type of model.
    Provides metrics to evaluate and visualize """
    def __init__(self, model_type: str, pipeline: object):
        self.model_type = model_type
        self.pipeline = pipeline

    def print_metrics(self, x_test, y_test, x_train, y_train):
        """Display various metrics to evaluate the performance of the model:
            -Accuracy
            -Precision
            -Recall
            -Confusion Matrix"""
        y_test = y_test.to_numpy()
        y_pred = self.pipeline.predict(x_test)
        y_pred_train = self.pipeline.predict(x_train)
        print(f"Model train accuracy : ", np.around(accuracy_score(y_train, y_pred_train), decimals=3))
        print("Model accuracy : ", np.around(accuracy_score(y_test, y_pred), decimals=3))
        print("Model precision : ", np.around(precision_score(y_test, y_pred, average="binary", pos_label=1), decimals=3))
        print("Model recall = ", np.around(recall_score(y_test, y_pred, average="binary", pos_label=1), decimals=3))
        print("Confusion Matrix : \n",
              confusion_matrix(y_test, y_pred, labels=[1, 0]))

        # TODO: put this in an interpretability object/function
        if self.model_type == "rf":
            print("Feature importance: \n",
                  np.around(self.pipeline.steps[1][1].feature_importances_, decimals=3))
        elif self.model_type == "lr":
            print("Feature coefficients: \n",
                  np.around(self.pipeline.steps[1][1].coef_, decimals=3))

    def plot_precision_recall(self, x_test, y_test, x_train, y_train, ColumnTransformer):
        """Get the engineered features and plot the precision recall curve for the model"""
        # get features
        transformer = ColumnTransformer
        transformer.fit(x_train, y_train)
        X_FE = transformer.transform(x_test)
        # get precision and recall
        y_score = self.pipeline.steps[1][1].predict_proba(X_FE)
        precision, recall, thresholds = precision_recall_curve(y_test, y_score[:, 1], pos_label=1)
        auc_precision_recall = auc(recall, precision)
        print("AUC precision recall curve is : ", np.around(auc_precision_recall, decimals=3))
        # plot results
        plt.plot(recall, precision, label='precision')
        thresholds = np.insert(thresholds, [0], 0)
        plt.plot(recall, thresholds, label='thresholds')
        plt.title(f'Precision recall curve for the {self.model_type} classifier')
        plt.xlabel('recall')
        plt.legend()
        plt.show()
