"""Plot train, valid, oot Precision Recall curves side-by-side"""

from matplotlib import pyplot as plt
from sklearn import metrics
from typing import Tuple


"""Models needed for plots"""

from collections import namedtuple

# Nametuple used for feeding values to plots.
SetValidation = namedtuple(
    "SetValidation", ["y_true", "y_pred", "title"]
)

def plot_precision_recall_vs_threshold(*tuples: Tuple[SetValidation]):  
    """Plot both precision and recall curves for different data
    sets (Typically: train, validation and test)

    Keyword Arguments:
        tuples {Tuple[SetValidation]} -- Different
            SetValidation instances, defining ground truths and
            predictions for different sets of data.

    Raises:
        ValueError: [description]
    """
    no_tuples = len(tuples)

    if no_tuples < 1:
        raise ValueError(
            "You shall not pass with no inputs"
        )
    
    if no_tuples <= 3:
        #fig, axes = plt.subplots(1, no_tuples) 
        fig, axes = plt.subplots(1, no_tuples, figsize=(12,4))
        for ax_i, tuple_i in zip(axes, tuples):
            true_i = tuple_i.y_true
            pred_i = tuple_i.y_pred
            title_i = tuple_i.title
            precisions_i, recalls_i, thresholds_i = \
                metrics.precision_recall_curve(true_i, pred_i)
            ax_i.plot(thresholds_i, precisions_i[:-1], "b--", label="Precision")
            ax_i.plot(thresholds_i, recalls_i[:-1], "g-", label="Recall")
            ax_i.set_ylabel("Score")
            ax_i.set_xlabel("Decision Threshold")
            ax_i.legend(loc='best')
            ax_i.grid()
            ax_i.set_title(title_i)

        fig.tight_layout()

    else:
        error_message = "You should not pass more than 3 sets yet."
        logger.error(error_message)
        raise ValueError(error_message)
        
        
# EXAMPLE FUNCTION CALL
nameYActual = 'fraudLong'
y_train_pred = trainDataAndPredictions['glm_p1'] #train fraud probability
y_test_pred = validDataAndPredictions['glm_p1']  #valid fraud probability
y_oot_pred = ootDataAndPredictions['glm_p1']     #oot fraud probability
train_set_validation = SetValidation(trainDataAndPredictions[nameYActual], y_train_pred, "Train")
valid_set_validation = SetValidation(validDataAndPredictions[nameYActual], y_test_pred, "Valid")
oot_set_validation = SetValidation(ootDataAndPredictions[nameYActual], y_oot_pred, "OOT")

# Make the precision/recall charts 
print("GLM MODEL, Trained on Fraud")  #title if you want one
plot_precision_recall_vs_threshold(train_set_validation, valid_set_validation, oot_set_validation)
