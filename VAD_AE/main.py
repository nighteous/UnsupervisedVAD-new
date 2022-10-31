import numpy as np
from train_ae import main



n_trials = 5


auc_list = []
for i in range(n_trials):
    print('trial: %d %s' % (i+1, '='*20))
    auc = main(max_epoch=20)
    auc_list.append(auc)

aucs = np.array(auc_list)

print('all aucs:', aucs)


print('AUC stats over %d runs : mean: %.2f, std: %.2f'% (n_trials, aucs.mean(), aucs.std()))

