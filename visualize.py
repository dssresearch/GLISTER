import matplotlib.pyplot as plt
import os
import pandas as pd
import sys
from utils.custom_dataset import *


def describe_data(x_trn, y_trn,num_cls,typ_d,path,logfile):
	com = np.append( np.reshape(y_trn, (-1, 1)), x_trn, axis=1)
	input_data = pd.DataFrame(com)

	header_names = ['F' + str(i) for i in range(input_data.shape[1]-1)]
	header_names.insert(0,'class')
	input_data.set_axis(header_names, axis=1, inplace=True)

	try:
		os.mkdir(path+"/boxplot", 0o777)
	except OSError:
	    print ("Directory %s exists" % path)

	for i in range(num_cls):
		temp = input_data.loc[input_data['class'] == float(i)]
		plt.figure();
		box = temp.loc[:, temp.columns != 'class'].boxplot()
		plt.savefig(path+"/boxplot/"+typ_d+str(i)+".svg", format="svg")
		plt.close()

	class_wise = input_data.groupby('class')		

	details = class_wise.describe()
	details.columns = details.columns.swaplevel(0, 1)

	details['count']['F0'].to_csv(logfile)

	for i in set(details.columns.get_level_values(0)):
		details[i].to_csv(logfile)

		fig = details[i].transpose().plot.line(title=data_name+" "+str(i)).get_figure()
		fig.savefig(path+"/"+typ_d+i+'.jpg')
		plt.close(fig)
	
datadir = sys.argv[1]
data_name = sys.argv[2]

output_dir = 'data_description/'+data_name
try:
    os.mkdir(output_dir, 0o777)
except OSError:
    print ("Directory %s exists" % output_dir)

if __name__ == "__main__": 

	path_logfile = os.path.join(output_dir, data_name + '.txt')
	logfile = open(path_logfile, 'w') 

	fullset, valset, testset, num_cls = load_dataset_numpy(datadir, data_name)

	x_trn, y_trn = fullset
	x_val, y_val = valset
	x_tst, y_tst = testset

	print("Training data",file=logfile)
	describe_data(x_trn, y_trn,num_cls,'train_',output_dir,logfile)
	print("Validation data",file=logfile)
	describe_data(x_val, y_val, num_cls,'val_',output_dir,logfile)
	print("Test data",file=logfile)
	describe_data(x_tst, y_tst,num_cls, 'test_',output_dir,logfile)



'''class_wise = df.groupby(0).median().transpose()

#class_wise.to_csv(str(fraction)+'_'+data_name+'.csv')

fig = class_wise.plot.line(title=str(fraction)+'_'+data_name).get_figure()
fig.savefig(str(fraction)+'_'+data_name+'_median.jpg')'''

''''''
