
from matplotlib import pyplot as plt

X = [0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.75]

#LR KNNSubmod cifar10_cont with val
Y1 = [0.7377, 0.7574, 0.7668, 0.7725, 0.7786, 0.7904, 0.8001]
Y2 = [0.7331, 0.7482, 0.7544, 0.7611, 0.766, 0.7846, 0.7938]

# KNN KNNSubmod cifar10_cont with val
#Y1 = [0.6037, 0.6232, 0.6344, 0.64, 0.6457, 0.6704, 0.6836]
#Y2 = [0.6017, 0.6197, 0.6328, 0.6316, 0.6456, 0.6587, 0.6779]

# LR KNNSubmod cifar100_cont with val
#Y1 = [0.431, 0.4391, 0.4478, 0.4474, 0.4558, 0.4887, 0.523]
#Y2 = [0.4322, 0.4452, 0.4475, 0.442, 0.4446, 0.4639, 0.4825]

# KNN KNNSubmod cifar100_cont with val
#Y1 = [0.3079, 0.3266, 0.342, 0.3549, 0.363, 0.3897, 0.4156]
#Y2 = [0.3065, 0.3276, 0.3404, 0.3459, 0.3564, 0.3797, 0.4001]


# LR KNNSubmod cifar10_bin with val
#Y1 = [0.7092, 0.7344, 0.7507, 0.7616, 0.7727, 0.7907, 0.8004]
#Y2 = [0.7156, 0.7337, 0.7557, 0.76, 0.7644, 0.78, 0.7865]

# KNN KNNSubmod cifar10_bin with val
#Y1 = [0.6115, 0.6225, 0.6303, 0.6355, 0.6428, 0.6626, 0.6759]
#Y2 = [0.6109, 0.6202, 0.6267, 0.6431, 0.6509, 0.6612, 0.6675]

# LR KNNSubmod cifar100_bin with val
#Y1 = [0.409, 0.4095, 0.4122, 0.4138, 0.4166, 0.4472, 0.479]
#Y2 = [0.4086, 0.4215, 0.4136, 0.4287, 0.424, 0.4302, 0.4486]

# KNN KNNSubmod cifar100_bin with val
#Y1 = [0.3074, 0.3295, 0.3453, 0.3571, 0.369, 0.3942]
#Y2 = [0.2984, 0.318, 0.3384, 0.3587, 0.3577, 0.3789]

# LR NBSubmod cifar10_bin with val
#Y1 = [0.7129, 0.7249, 0.7549, 0.7664, 0.772, 0.7881, 0.7998]
#Y2 = [0.7198, 0.7352, 0.7573, 0.7597, 0.7645, 0.7825, 0.7881]

# KNN NBSubmod cifar10_bin with val
#Y1 = [0.6072, 0.6214, 0.6302, 0.6372, 0.6474, 0.669, 0.6799]
#Y2 = [0.616, 0.621, 0.6269, 0.6407, 0.6449, 0.662, 0.6742]


# LR KNNSubmod cifar10_cont with val = trn
#Y1 = [0.7278, 0.7446, 0.7568, 0.7684, 0.7745, 0.79, 0.8019]
#Y2 = [0.7211, 0.742, 0.7574, 0.7595, 0.7723, 0.781, 0.7859]

# KNN KNNSubmod cifar10_cont with val = trn
#Y1 = [0.6607, 0.6635, 0.6664, 0.6683, 0.6706, 0.6796, 0.6867]
#Y2 = [0.6129, 0.6132, 0.6347, 0.6365, 0.6525, 0.66, 0.6752]

# LR KNNSubmod cifar100_cont with val = trn
#Y1 = [0.4458, 0.4423, 0.448, 0.4533, 0.4719, 0.5036, 0.5393]
#Y2 = [0.4439, 0.446, 0.4554, 0.458, 0.4707, 0.4911, 0.5042]

# KNN KNNSubmod cifar100_cont with val = trn
#Y1 = [0.3824, 0.3873, 0.3905, 0.393, 0.3949, 0.4064, 0.4189]
#Y2 = [0.3051, 0.3289, 0.3331, 0.3438, 0.3552, 0.3805, 0.3983]

F = plt.gcf()
Size = F.get_size_inches()
F.set_size_inches(Size[0]*2, Size[1]*2, forward=True)

plt.plot(X, Y1, '-r', label='Submod Subset Sel', marker='o', markerfacecolor ='g', markeredgecolor ='g', markersize=3)
plt.plot(X, Y2, '-b', label='Random Subset Sel',marker='s', markerfacecolor ='g', markeredgecolor ='g', markersize=3)
plt.legend(loc="upper left")
plt.xlabel('Budgets')
plt.ylabel('Accuracy')
plt.title('KNN with KNNSubmod selection on CIFAR-100 Binned with val')
for a,b in zip(X, Y1): 
    plt.text(a, b, str(b), fontsize=8, bbox=dict(facecolor='red', alpha=0.5))
        
for a,b in zip(X, Y2): 
    plt.text(a, b, str(b), fontsize=8, bbox=dict(facecolor='blue', alpha=0.5))
                
plt.savefig('KNN_KNNSubmod_cifar100_bin.png')
