import numpy as np
import matplotlib.pyplot as plt


def get_auc(fpr, tpr):
    auc = []
    for i in range(len(fpr)):
        auc_i = 0.0
        fpr_i = fpr[i]
        tpr_i = tpr[i]
        for j in range(len(fpr_i) - 1):
            delta_x = abs(fpr_i[j] - fpr_i[j + 1])
            delta_y = (tpr_i[j] + tpr_i[j + 1]) * 0.5
            auc_i += delta_x * delta_y
        auc.append(auc_i)
    return auc


def ood_aupr_all(dataset):  # last figure 01/27
    # 0 vacuity
    # 1 Dissonance
    # 2 Aleatoric
    # 3 Epistemic
    # 4 Entropy
    # 5 Distribution
    # 6 GCN_Entropy
    # 7 Dropout Entropy
    # 9 Dropout Aleatoric
    # 9 Dropout Epistemic
    plt.style.use('seaborn-whitegrid')  # seaborn-whitegrid, ggplot
    recall = np.load("data/ood/recall_{}_ood.npy".format(dataset))
    precision = np.load("data/ood/precision_{}_ood.npy".format(dataset))
    x = np.mean(recall, axis=0)
    y = np.mean(precision, axis=0)
    xx = [0, 1]
    yy = [y[0][-1], y[0][-1]]
    linewidth_ = 2

    plt.plot(xx, yy, linewidth=linewidth_, color='k', linestyle='-.', marker=',', label='Random', ms=7)
    # plt.plot(x[6], y[6], linewidth=linewidth_, color='y', linestyle='--', marker=',', label='GCN Entropy', ms=7)
    # plt.plot(x[7], y[7], linewidth=linewidth_, color='blueviolet', linestyle='--', marker=',', label='Dropout Entropy',
    #          ms=7)
    # plt.plot(x[8], y[8], linewidth=linewidth_, color='lightcoral', linestyle='--', marker=',',
    #          label='Dropout Aleatoric', ms=7)
    # plt.plot(x[9], y[9], linewidth=linewidth_, color='darkseagreen', linestyle='--', marker=',',
    #          label='Dropout Epistemic', ms=7)
    plt.plot(x[5], y[5], linewidth=linewidth_, color='orchid', linestyle='-', marker=',', label='Diff. Entropy', ms=7)
    plt.plot(x[4], y[4], linewidth=linewidth_, color='gold', linestyle='-', marker=',', label='Entropy',
             ms=7)  # slateblue, gold
    plt.plot(x[2], y[2], linewidth=linewidth_, color='darkorange', linestyle='-', marker=',', label='Aleatoric', ms=7)
    plt.plot(x[3], y[3], linewidth=linewidth_, color='lightseagreen', linestyle='-', marker=',', label='Epistemic',
             ms=7)
    plt.plot(x[1], y[1], linewidth=linewidth_, color='royalblue', linestyle='-', marker=',', label='Dissonance', ms=7)
    plt.plot(x[0], y[0], linewidth=linewidth_, color='tomato', linestyle='-', marker=',', ms=7, label='Vacuity')

    # plt.plot(x[4], y[4], linewidth=2, color='slateblue', linestyle='-', marker=',', label='Total', ms=7)
    # plt.plot(x[5], y[5], linewidth=2, color='lightseagreen', linestyle='--', marker=',', label='Epistemic var', ms=7)
    # plt.plot(x[6], y[6], linewidth=2, color='lightseagreen', linestyle='-.', marker=',', label='Epistemic var2',ms=7)

    plt.legend(loc='best', fontsize=8, ncol=2)
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)

    # if dataset == 'cora':
    #     plt.ylim((0.1, 1.0))
    #     my_y_ticks = [0.2, 0.4, 0.6, 0.8, 1.0]
    #     plt.yticks(my_y_ticks)
    # if dataset == 'citeseer':
    #     plt.ylim((0.3, 0.9))
    #     my_y_ticks = [0.4, 0.6, 0.8, 1.0]
    #     plt.yticks(my_y_ticks)
    # if dataset == 'pubmed':
    #     plt.ylim((0.2, 1.0))
    #     my_y_ticks = [0.2, 0.4, 0.6, 0.8, 1.0]
    #     plt.yticks(my_y_ticks)
    # plt.yticks(fontsize=14)

    # plt.title(dataset)
    plt.savefig("data/fig/aupr_{}_ood.png".format(dataset), dpi=500)
    plt.show()
    # AUC = get_auc(recall, precision)
    AUC = get_auc(recall, precision)
    return AUC


def ood_auroc_all(dataset):  # final figure 01/27
    # 0 vacuity
    # 1 Dissonance
    # 2 Aleatoric
    # 3 Epistemic
    # 4 Entropy
    # 5 Distribution
    # 6 GCN_Entropy
    # 7 Dropout Entropy
    # 9 Dropout Aleatoric
    # 9 Dropout Epistemic
    threshold = 0.01
    plt.style.use('seaborn-whitegrid')  # seaborn-whitegrid, ggplot

    fpr = np.load("data/ood/fpr_{}_ood.npy".format(dataset))
    tpr = np.load("data/ood/tpr_{}_ood.npy".format(dataset))
    x = np.mean(fpr, axis=0)
    y = np.mean(tpr, axis=0)
    xx = [0.0, 1]
    # if dataset == 'cora':
    #     y[0] = y[0] + 0.003
    linewidth_ = 2
    plt.plot(xx, xx, linewidth=linewidth_, color='k', linestyle='-.', marker=',', label='Random', ms=7)
    # plt.plot(x[6], y[6], linewidth=linewidth_, color='royalblue', linestyle='--', marker=',', label='GCN Entropy',
    #          ms=7)  # y
    # plt.plot(x[7], y[7], linewidth=linewidth_, color='gold', linestyle='--', marker=',', label='Dropout Entropy',
    #          ms=7)  # blueviolet
    # plt.plot(x[9], y[9], linewidth=linewidth_, color='darkorange', linestyle='--', marker=',',
    #          label='Dropout Epistemic', ms=7)  # darkseagreen
    # plt.plot(x[8], y[8], linewidth=linewidth_, color='lightseagreen', linestyle='--', marker=',',
    #          label='Dropout Aleatoric', ms=7)  # lightcoral
    # plt.plot(x[5], y[5], linewidth=linewidth_, color='orchid', linestyle='--', marker=',', label='Diff. Entropy', ms=7)
    # plt.plot(x[4], y[4], linewidth=linewidth_, color='gold', linestyle='-', marker=',', label='Entropy',
    #          ms=7)  # slateblue, gold
    # plt.plot(x[3], y[3], linewidth=linewidth_, color='darkorange', linestyle='-', marker=',', label='Epistemic',
    #          ms=7)  # lightseagreen
    # plt.plot(x[2], y[2], linewidth=linewidth_, color='lightseagreen', linestyle='-', marker=',', label='Aleatoric',
    #          ms=7)  # darkorange
    # plt.plot(x[1], y[1], linewidth=linewidth_, color='royalblue', linestyle='-', marker=',', label='Dissonance', ms=7)
    # plt.plot(x[0], y[0], linewidth=linewidth_, color='tomato', linestyle='-', marker=',', ms=7, label='Vacuity')
    plt.plot(x[5], y[5], linewidth=linewidth_, color='orchid', linestyle='-', marker=',', label='Diff. Entropy', ms=7)
    plt.plot(x[4], y[4], linewidth=linewidth_, color='gold', linestyle='-', marker=',', label='Entropy',
             ms=7)  # slateblue, gold
    plt.plot(x[2], y[2], linewidth=linewidth_, color='darkorange', linestyle='-', marker=',', label='Aleatoric', ms=7)
    plt.plot(x[3], y[3], linewidth=linewidth_, color='lightseagreen', linestyle='-', marker=',', label='Epistemic',
             ms=7)
    plt.plot(x[1], y[1], linewidth=linewidth_, color='royalblue', linestyle='-', marker=',', label='Dissonance', ms=7)
    plt.plot(x[0], y[0], linewidth=linewidth_, color='tomato', linestyle='-', marker=',', ms=7, label='Vacuity')

    plt.legend(loc='best', fontsize=8, ncol=2)
    plt.xlabel('False Positive Rate', fontsize=15)
    plt.ylabel('True Positive Rate', fontsize=15)

    # if dataset == 'cora':
    #     plt.ylim((0.1, 1.0))
    #     my_y_ticks = [0.2, 0.4, 0.6, 0.8, 1.0]
    #     plt.yticks(my_y_ticks)
    # if dataset == 'citeseer':
    #     plt.ylim((0.3, 0.9))
    #     my_y_ticks = [0.4, 0.6, 0.8, 1.0]
    #     plt.yticks(my_y_ticks)
    # if dataset == 'pubmed':
    #     plt.ylim((0.2, 1.0))
    #     my_y_ticks = [0.2, 0.4, 0.6, 0.8, 1.0]
    #     plt.yticks(my_y_ticks)

    plt.ylim((-0.00, 1.01))
    plt.xlim((-0.00, 1.0))

    # plt.title(dataset)
    plt.savefig("data/fig/auroc_{}_ood.png".format(dataset), dpi=500)
    plt.show()
    AUC = get_auc(fpr, tpr)
    return AUC


if __name__ == '__main__':
    datasets = ['cora', 'citeseer', 'pubmed', 'amazon_electronics_photo', 'amazon_electronics_computers', 'ms_academic_phy']
    for dataset in datasets:
        ood_aupr_all(dataset)
        ood_auroc_all(dataset)
