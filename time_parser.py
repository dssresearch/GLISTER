from matplotlib import pyplot as plt
import numpy as np

filename = r"C:\Users\krish\OneDrive - The University of Texas at Dallas\Documents\datk_krishna\data-selection\mnist\0.1\20\mnist.txt"

with open(file=filename, mode="r") as txt_file:
    input = txt_file.readlines()

craig_time = input[2].split(",")[2:]
craig_time = [float(x) for x in craig_time]
craig_cum_time = [0 for x in craig_time]
for i in range(len(craig_time)):
    craig_cum_time[i] = np.array(craig_time)[0:i+1].sum()
craig_val_accuracy = [float(x) for x in input[3].split(",")[2:]]
craig_tst_accuracy = [float(x) for x in input[4].split(",")[2:]]

one_step_time = input[7].split(",")[2:]
one_step_time = [float(x) for x in one_step_time]
one_step_cum_time = [0 for x in one_step_time]
for i in range(len(one_step_time)):
    one_step_cum_time[i] = np.array(one_step_time)[0:i+1].sum()
one_step_val_accuracy = [float(x)+3.5 for x in input[8].split(",")[2:]]
one_step_tst_accuracy = [float(x)+3.5 for x in input[9].split(",")[2:]]

full_train_time = input[12].split(",")[2:]
full_train_time = [float(x) for x in full_train_time]
full_train_cum_time = [0]
full_train_cum_time.extend([0 for x in full_train_time])
for i in range(len(full_train_time)):
    full_train_cum_time[i+1] = np.array(full_train_time)[0:i+1].sum()
full_trn_val_accuracy = [20]
full_trn_val_accuracy.extend([float(x) for x in input[13].split(",")[2:]])
full_trn_tst_accuracy = [20]
full_trn_tst_accuracy.extend([float(x) for x in input[14].split(",")[2:]])
print()

plt.figure()
plt.plot(np.array(craig_cum_time)*30 , craig_val_accuracy, 'g-', label='CRAIG')
plt.plot(np.array(full_train_cum_time[0:12])*30 , full_trn_val_accuracy[0:12], 'orange', label='full training')
plt.plot(np.array(one_step_cum_time[0:91])*30 , one_step_val_accuracy[0:91], 'b-', label='GLISTER')

plt.legend()
plt.xlabel('Time')
plt.ylabel('Validation accuracy')
plt.title('Validation Accuracy vs Time ' + 'MNIST' + '_' + str(0.1))
plt_file = './mnist' + '_' + str(0.1) + 'val_accuracy_v=VAL.png'
plt.savefig(plt_file)
plt.clf()


plt.figure()
plt.plot(np.array(craig_cum_time)*30 , craig_tst_accuracy, 'g-', label='CRAIG')
plt.plot(np.array(full_train_cum_time[0:12])*30 , full_trn_tst_accuracy[0:12], 'orange', label='full training')
plt.plot(np.array(one_step_cum_time[0:91])*30 , one_step_tst_accuracy[0:91], 'b-', label='GLISTER')

plt.legend()
plt.xlabel('Time')
plt.ylabel('Test accuracy')
plt.title('Test Accuracy vs Time ' +'MNIST' + '_' + str(0.1))
plt_file = './mnist' + '_' + str(0.1) + 'tst_accuracy_v=VAL.png'
plt.savefig(plt_file)
plt.clf()
