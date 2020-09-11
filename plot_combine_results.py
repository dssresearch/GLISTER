import numpy as np
import matplotlib.pyplot as plt 
import os
import sys

directory = sys.argv[1]
file = os.path.join(directory,sys.argv[2])

sel = 0
acc =[]
x_axis =[]

colors = ['g-','c-','b-','r-','orange','#000000','#8c564b','m-','y','pink']

with open(file) as fp:
   line = fp.readline()
   data_name = line.strip()
   line = fp.readline()
   while line:
   		if len(line) < 2:
   			
   			if len(acc) > 0:

   				acc = np.array(acc)
   				#print(len(labels))
   				for i in range(len(labels)):
   					plt.plot(x_axis, acc[:,i],colors[i],label = labels[i])

   				plt.legend() 
   				plt.xlabel('Budget') 
   				plt.ylabel(typeOf+' Accuracy')
   				plt.title(typeOf+'_Accuracy_'+data_name+"_sel_"+sel) 

   				plt.savefig(directory+"/"+typeOf+'_Accuracy_'+data_name+"_sel_"+sel+'.png')
   				plt.clf()

   				acc = []
   				x_axis =[]

   			line = fp.readline()
   			continue

   		if line[0] != '|':
   			info = line.strip().split(" ")
   			if info[0] == "Select":
   				sel = info[-1]
   			elif info[0] in ["Validation","Test"]:
   				typeOf =  info[0]
   		else:
   			data = line[1:].strip()[:-1].split("|")

   			if data[0] == "Subset Size (%)":
   				labels = data[1:]

   			else:
   				x_axis.append(int(data[0]))
   				acc.append([float(i) for i in data[1:]])

   		line = fp.readline()

if len(acc) > 0:

	acc = np.array(acc)
	#print(len(labels))
	for i in range(len(labels)):
		plt.plot(x_axis, acc[:,i],colors[i],label = labels[i])

	plt.legend() 
	plt.xlabel('Budget') 
	plt.ylabel(typeOf+' Accuracy')
	plt.title(typeOf+'_Accuracy_'+data_name+"_sel_"+sel) 

	plt.savefig(directory+"/"+typeOf+'_Accuracy_'+data_name+"_sel_"+sel+'.png')
	plt.clf()