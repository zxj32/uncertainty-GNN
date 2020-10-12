from metrics import *
import random
from Load_npz import load_npz_data2, load_npz_data_ood_train2
from scipy.special import loggamma, digamma
from utils import load_data_threshold, load_data_ood
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score


def vacuity_uncertainty(Baye_result):
    # Vacuity uncertainty
    mean = np.mean(Baye_result, axis=0)
    class_num = mean.shape[1]
    alpha = np.exp(mean) + 1.0
    S = np.sum(alpha, axis=1, keepdims=True)
    un_vacuity = class_num / S

    return un_vacuity


def vacuity_sgcn(mean):
    # Vacuity uncertainty
    class_num = mean.shape[1]
    alpha = np.exp(mean) + 1.0
    S = np.sum(alpha, axis=1, keepdims=True)
    un_vacuity = class_num / S

    return un_vacuity

def dissonance_uncertainty(Baye_result):
    mean = np.mean(Baye_result, axis=0)
    evidence = np.exp(mean)
    alpha = np.exp(mean) + 1.0
    S = np.sum(alpha, axis=1, keepdims=True)
    belief = evidence / S
    dis_un = np.zeros_like(S)
    for k in range(belief.shape[0]):
        for i in range(belief.shape[1]):
            bi = belief[k][i]
            term_Bal = 0.0
            term_bj = 0.0
            for j in range(belief.shape[1]):
                if j != i:
                    bj = belief[k][j]
                    term_Bal += bj * Bal(bi, bj)
                    term_bj += bj
            dis_ki = bi * term_Bal / term_bj
            dis_un[k] += dis_ki

    return dis_un


def dissonance_sgcn(mean):
    evidence = np.exp(mean)
    alpha = np.exp(mean) + 1.0
    S = np.sum(alpha, axis=1, keepdims=True)
    belief = evidence / S
    dis_un = np.zeros_like(S)
    for k in range(belief.shape[0]):
        for i in range(belief.shape[1]):
            bi = belief[k][i]
            term_Bal = 0.0
            term_bj = 0.0
            for j in range(belief.shape[1]):
                if j != i:
                    bj = belief[k][j]
                    term_Bal += bj * Bal(bi, bj)
                    term_bj += bj
            dis_ki = bi * term_Bal / term_bj
            dis_un[k] += dis_ki

    return dis_un

def Bal(b_i, b_j):
    result = 1 - np.abs(b_i - b_j) / (b_i + b_j)
    return result


def entropy_SL(mean):
    class_num = mean.shape[1]
    alpha = np.exp(mean) + 1.0
    S = np.sum(alpha, axis=1, keepdims=True)
    prob = alpha / S
    entropy = - prob * (np.log(prob) / np.log(class_num))
    total_un = np.sum(entropy, axis=1, keepdims=True)
    class_un = entropy
    return total_un, class_un


def entropy(pred):
    class_num = pred.shape[1]
    prob = pred
    entropy = - prob * (np.log(prob) / np.log(class_num))
    total_un = np.sum(entropy, axis=1, keepdims=True)
    class_un = entropy
    return total_un, class_un


def entropy_softmax(pred):
    class_num = pred.shape[1]
    prob = softmax(pred)
    entropy = - prob * (np.log(prob) / np.log(class_num))
    total_un = np.sum(entropy, axis=1, keepdims=True)
    class_un = entropy
    return total_un, class_un


def entropy_dropout(pred):
    mean = []
    for p in pred:
        prob_i = softmax(p)
        mean.append(prob_i)
    mean = np.mean(mean, axis=0)
    class_num = mean.shape[1]
    prob = mean
    entropy = - prob * (np.log(prob) / np.log(class_num))
    total_un = np.sum(entropy, axis=1, keepdims=True)
    class_un = entropy
    return total_un, class_un


def aleatoric_dropout(Baye_result):
    al_un = []
    al_class_un = []
    for item in Baye_result:
        un, class_un = entropy_softmax(item)
        al_un.append(un)
        al_class_un.append(class_un)
    ale_un = np.mean(al_un, axis=0)
    ale_class_un = np.mean(al_class_un, axis=0)
    return ale_un, ale_class_un


def softmax(pred):
    ex = np.exp(pred - np.amax(pred, axis=1, keepdims=True))
    prob = ex / np.sum(ex, axis=1, keepdims=True)
    return prob


def total_uncertainty(Baye_result):
    prob_all = []
    class_num = Baye_result[0].shape[1]
    for item in Baye_result:
        alpha = np.exp(item) + 1.0
        S = np.sum(alpha, axis=1, keepdims=True)
        prob = alpha / S
        prob_all.append(prob)
    prob_mean = np.mean(prob_all, axis=0)
    total_class_un = - prob_mean * (np.log(prob_mean) / np.log(class_num))
    total_un = np.sum(total_class_un, axis=1, keepdims=True)
    return total_un, total_class_un


def aleatoric_uncertainty(Baye_result):
    al_un = []
    al_class_un = []
    for item in Baye_result:
        un, class_un = entropy_SL(item)
        al_un.append(un)
        al_class_un.append(class_un)
    ale_un = np.mean(al_un, axis=0)
    ale_class_un = np.mean(al_class_un, axis=0)
    return ale_un, ale_class_un


def dpn_epistemic(result):
    alpha = np.exp(result) + 1.0
    S = np.sum(alpha, axis=1, keepdims=True)
    p = alpha/S
    term1 = np.log(p) - digamma(alpha + 1) + digamma(S + 1)
    un = p * term1
    un = -np.sum(un, axis=1, keepdims=True)
    return un

def get_un_dpn(result):
    epi = dpn_epistemic(result)
    ent, _ = entropy_SL(result)
    alea = ent - epi
    uncertainty = []
    uncertainty.append(alea)
    uncertainty.append(epi)
    uncertainty.append(ent)
    return uncertainty

def get_un_sgcn(result):
    ent, _ = entropy_SL(result)
    vac = vacuity_sgcn(result)
    diss = dissonance_sgcn(result)
    uncertainty = []
    uncertainty.append(vac)
    uncertainty.append(diss)
    uncertainty.append(ent)
    return uncertainty

def get_un_EDL(result):
    ent, _ = entropy_SL(result)
    vac = vacuity_sgcn(result)
    diss = dissonance_sgcn(result)
    uncertainty = []
    uncertainty.append(vac)
    uncertainty.append(diss)
    uncertainty.append(ent)
    return uncertainty

def get_uncertainty(Baye_result):
    uncertainty = []
    uncertainty_class = []
    un_vacuity = vacuity_uncertainty(Baye_result)
    un_dissonance = dissonance_uncertainty(Baye_result)
    un_total, un_total_class = total_uncertainty(Baye_result)
    un_aleatoric, un_aleatoric_class = aleatoric_uncertainty(Baye_result)
    # un_epistemic = un_total - un_aleatoric
    un_epistemic_class = un_total_class - un_aleatoric_class
    un_epistemic = np.sum(un_epistemic_class, axis=1, keepdims=True)
    un_var3_class = np.var(Baye_result, axis=0)
    un_var3 = np.sum(un_var3_class, axis=1, keepdims=True)
    diff_en = diff_entropy(Baye_result)
    uncertainty.append(un_vacuity)
    uncertainty.append(un_dissonance)
    uncertainty.append(un_aleatoric)
    uncertainty.append(un_epistemic)
    # uncertainty.append(diff_en)
    uncertainty.append(un_total)
    # uncertainty.append(un_ep_var)
    # uncertainty.append(un_ep_var2)
    # uncertainty.append(un_var3)
    uncertainty_class.append(un_aleatoric_class)
    uncertainty_class.append(un_epistemic_class)
    uncertainty_class.append(un_total_class)
    return uncertainty


def diff_entropy(Baye_result):
    mean = np.mean(Baye_result, axis=0)
    alpha = np.exp(mean) + 1.0
    S = np.sum(alpha, axis=1, keepdims=True)
    ln_gamma = loggamma(alpha)
    ln_gamma_S = loggamma(S)
    term1 = np.sum(ln_gamma, axis=1, keepdims=True) - ln_gamma_S
    digmma_alpha = digamma(alpha)
    digamma_S = digamma(S)
    term2 = (alpha - 1) * (digmma_alpha - digamma_S)
    term2 = np.sum(term2, axis=1, keepdims=True)
    diff_en = term1 - term2

    return diff_en


def get_un_dropout(pred):
    un = []
    dr_entroy, dr_entroy_class = entropy_dropout(pred)
    dr_ale, dr_ale_clsss = aleatoric_dropout(pred)
    dr_eps_class = dr_entroy_class - dr_ale_clsss
    dr_eps = np.sum(dr_eps_class, axis=1, keepdims=True)
    un.append(dr_entroy)
    un.append(dr_ale)
    un.append(dr_eps)
    return un

def get_un_entropy(pred):
    un = []
    dr_entroy, dr_entroy_class = entropy_softmax(pred)
    un.append(dr_entroy)

    return un


def Misclassification_npz(output, dataset, model):  ## table 2
    _, _, _, _, y_test, train_mask, _, test_mask, labels, test_idx = load_npz_data2(dataset, 223)

    if model == "S_GCN":
        uncertainties = get_un_sgcn(output)
        mean = output
    elif model == "S_BGCN_T" or model == "S_BGCN_T_K" or model == "S_BGCN":
        uncertainties = get_uncertainty(output)
        mean = np.mean(output, axis=0)
    elif model == "DPN":
        uncertainties = get_un_dpn(output)
        mean = output
    elif model == "EDL":
        uncertainties = get_un_EDL(output)
        mean = output
    elif model == "Drop":
        uncertainties = get_un_dropout(output)
        mean = np.mean(output, axis=0)
    elif model == "GCN":
        uncertainties = get_un_entropy(output)
        mean = output

    prediction = np.equal(np.argmax(mean, 1), np.argmax(labels, 1))
    auroc_s = []
    aupr_s = []

    random.seed(123)
    test_index = []
    test_idx = list(test_idx)
    for i in range(10):
        test_index_i = random.sample(test_idx, 1000)
        test_index.append(test_index_i)

    for index in test_index:
        prediction_i = prediction[index]
        un_roc = []
        un_pr = []
        for uncertainty in uncertainties:
            un_i = uncertainty[index]
            un_roc.append(roc_auc_score(prediction_i, -np.array(un_i)))
            un_pr.append(average_precision_score(prediction_i, -np.array(un_i)))
        auroc_s.append(un_roc)
        aupr_s.append(un_pr)
    return np.mean(auroc_s, axis=0), np.mean(aupr_s, axis=0)

def Misclassification(output, dataset, model):  ## table 2
    _, _, _, _, y_test, train_mask, _, test_mask, labels = load_data_threshold(dataset)

    if model == "S_GCN":
        uncertainties = get_un_sgcn(output)
        mean = output
    elif model == "S_BGCN_T" or model == "S_BGCN_T_K" or model == "S_BGCN":
        uncertainties = get_uncertainty(output)
        mean = np.mean(output, axis=0)
    elif model == "DPN":
        uncertainties = get_un_dpn(output)
        mean = output
    elif model == "EDL":
        uncertainties = get_un_EDL(output)
        mean = output
    elif model == "Drop":
        uncertainties = get_un_dropout(output)
        mean = np.mean(output, axis=0)
    elif model == "GCN":
        uncertainties = get_un_entropy(output)
        mean = output

    prediction = np.equal(np.argmax(mean, 1), np.argmax(labels, 1))
    train_num = np.sum(train_mask)
    test_index = []
    auroc_s = []
    aupr_s = []
    for i in range(10):
        test_index_i = random.sample(range(int(train_num), len(test_mask)), 1000)
        test_index.append(test_index_i)
    for index in test_index:
        prediction_i = prediction[index]
        un_roc = []
        un_pr = []
        for uncertainty in uncertainties:
            un_i = uncertainty[index]
            un_roc.append(roc_auc_score(prediction_i, -np.array(un_i)))
            un_pr.append(average_precision_score(prediction_i, -np.array(un_i)))
        auroc_s.append(un_roc)
        aupr_s.append(un_pr)
    return np.mean(auroc_s, axis=0), np.mean(aupr_s, axis=0)


def OOD_Detection_npz(output, dataset, model):  ## table 3
    _, _, _, _, _, _, test_mask, idx_train = load_npz_data_ood_train2(dataset, 223)
    if model == "S_GCN":
        uncertainties = get_un_sgcn(output)
    elif model == "S_BGCN_T" or model == "S_BGCN_T_K" or model == "S_BGCN":
        uncertainties = get_uncertainty(output)
    elif model == "DPN":
        uncertainties = get_un_dpn(output)
    elif model == "EDL":
        uncertainties = get_un_EDL(output)
    elif model == "Drop":
        uncertainties = get_un_dropout(output)
    elif model == "GCN":
        uncertainties = get_un_entropy(output)

    prediction = test_mask

    auroc_s = []
    aupr_s = []
    random.seed(123)
    test_idx = list(range(len(test_mask)))
    for x in idx_train:
        test_idx.remove(x)
    test_index = []
    for i in range(10):
        test_index_i = random.sample(test_idx, 1000)
        test_index.append(test_index_i)
    for index in test_index:
        prediction_i = prediction[index]
        un_roc = []
        un_pr = []
        for uncertainty in uncertainties:
            un_i = uncertainty[index]
            un_roc.append(roc_auc_score(prediction_i, np.array(un_i)))
            un_pr.append(average_precision_score(prediction_i, np.array(un_i)))
        auroc_s.append(un_roc)
        aupr_s.append(un_pr)
    return np.mean(auroc_s, axis=0), np.mean(aupr_s, axis=0)

def OOD_Detection(output, dataset, model):  ## table 3
    _, _, _, _, _, _, test_mask_all, test_mask = load_data_ood(dataset)
    if model == "S_GCN":
        uncertainties = get_un_sgcn(output)
    elif model == "S_BGCN_T" or model == "S_BGCN_T_K" or model == "S_BGCN":
        uncertainties = get_uncertainty(output)
    elif model == "DPN":
        uncertainties = get_un_dpn(output)
    elif model == "EDL":
        uncertainties = get_un_EDL(output)
    elif model == "Drop":
        uncertainties = get_un_dropout(output)
    elif model == "GCN":
        uncertainties = get_un_entropy(output)

    test_num_all = np.sum(test_mask_all)
    prediction = test_mask
    train_num = len(test_mask_all) - test_num_all

    test_index = []
    auroc_s = []
    aupr_s = []
    for i in range(10):
        test_index_i = random.sample(range(int(train_num), len(test_mask)), 1000)
        test_index.append(test_index_i)
    for index in test_index:
        prediction_i = prediction[index]
        un_roc = []
        un_pr = []
        for uncertainty in uncertainties:
            un_i = uncertainty[index]
            un_roc.append(roc_auc_score(prediction_i, np.array(un_i)))
            un_pr.append(average_precision_score(prediction_i, np.array(un_i)))
        auroc_s.append(un_roc)
        aupr_s.append(un_pr)
    return np.mean(auroc_s, axis=0), np.mean(aupr_s, axis=0)

