import numpy as np


def ECE(all_probs, ytest, num_bins = 1, print_ = True):
    # Compute the Expected Calibration Error (ECE)
    prob_preds = np.max(all_probs, 1)
    predictions = np.argmax(all_probs, 1)
    accuracies = (predictions == ytest)
    ece = 0.
    for it in range(num_bins):
        ind_bin = (it / num_bins < prob_preds) * (prob_preds <= (it + 1) / num_bins)
        if not ind_bin.any():
            continue
        acc_bin = accuracies[ind_bin]
        prob_bin = prob_preds[ind_bin]
        ece_bin = ind_bin.sum() * np.abs(acc_bin.mean() - prob_bin.mean())
        ece += ece_bin
    if print_:
        print("ECE =", ece / len(ytest))
    return ece / len(ytest)


def BS(ytest, all_probs, print_ = True):
    num_classes = len(np.unique(ytest))
    # Perform a one-hot encoding
    labels_true = np.eye(num_classes)[ytest]
    # Compute the Brier Score (BS)
    bs = num_classes * np.mean((all_probs - labels_true) ** 2)
    if print_:
        print("BS =", bs)
    return bs


def NLL(all_probs, ytest, print_ = True):
    # Compute the Negative Log Likelihood (NLL)
    log_it = - np.log(np.take_along_axis(all_probs, np.expand_dims(ytest, axis=1), axis=1)).squeeze()
    nll = log_it.mean()
    if print_:
        print(f"NNL = {nll}")
    return nll
