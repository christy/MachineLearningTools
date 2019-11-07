import collections
from collections import namedtuple

# Nametuple used for feeding values to plots.
SetValidation = collections.namedtuple(
    "SetValidation", ["y_true", "y_pred", "title"]
)

"""Module to help when building classification related plot"""


def plot_precision_recall_vs_threshold(*tuples: SetValidation, posLabel):
    """Plot both precision and recall curves for different data
    sets (Typically: train, validation and test)
    """
    no_tuples = len(tuples)

    if no_tuples < 1:
        raise ValueError(
            "You shall not pass with no inputs"
        )

    if no_tuples <= 3:
        fig, axes = plt.subplots(1, no_tuples)
        for ax_i, tuple_i in zip(axes, tuples):
            true_i = tuple_i.y_true
            pred_i = tuple_i.y_pred
            title_i = tuple_i.title
            precisions_i, recalls_i, thresholds_i = \
                metrics.precision_recall_curve(true_i, pred_i, pos_label=posLabel)
            ax_i.plot(
                thresholds_i, precisions_i[:-1], "b--", label="Precision"
            )
            ax_i.plot(thresholds_i, recalls_i[:-1], "g-", label="Recall")
            ax_i.set_ylabel("Score")
            ax_i.set_xlabel("Decision Threshold")
            ax_i.legend(loc='best')
            ax_i.grid()
            ax_i.set_title(title_i)

        fig.tight_layout()

    else:
        error_message = "You should not pass more than 3 sets yet."
        #logger.error(error_message)
        raise ValueError(error_message)


def plot_rocs(*tuples: SetValidation, posLabel):
    """Plots ROC curves for multiple sets of data.
    """
    no_tuples = len(tuples)

    if no_tuples < 1:
        raise ValueError(
            "You shall not pass with no inputs"
        )

    if no_tuples <= 3:
        plt.figure()

        for tuple_i in tuples:
            fpri, tpri, thi = metrics.roc_curve(tuple_i.y_true
                                                , tuple_i.y_pred
                                                , pos_label=posLabel)
            auci = metrics.auc(fpri, tpri)
            label_i = "{}: {:0.4f}".format(
                tuple_i.title, auci
                
            )
            plt.plot(fpri, tpri, linewidth=2, label=label_i)

        plt.title('ROC Curve')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.axis([-0.005, 1, 0, 1.005])
        plt.xticks(np.arange(0, 1, 0.05), rotation=90)

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate (Recall)")
        plt.legend(loc='best')
        plt.grid()
        plt.tight_layout()

    else:
        error_message = "You should not pass more than 3 sets yet."
        logger.error(error_message)
        raise ValueError(error_message)
