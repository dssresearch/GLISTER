import numpy as np
import os
import sys
import copy

#[dna,'sklearn-digits','satimage','svmguide1','letter','shuttle','ijcnn1','sensorless','connect_4','sensit_seismic']

data_name = sys.argv[2]
directory = sys.argv[1]

in_dir = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]

fractions = [ int(100*round(float(i), 2)) for i in in_dir]

fractions, in_dir = zip(*sorted(zip(fractions, in_dir)))

selection =[]
for di in in_dir:
	temp = [int(i) for i in os.listdir(os.path.join(directory, di))]
	if len(selection) < len(temp): 
		selection = copy.deepcopy(temp)

selection.sort()

if data_name == 'mnist':
	method = ["FacLoc","One - Step Taylor", "Random","Random Class Prior ","Random Online ","CRAIG",\
"Full Data Training","Rand-Reg One-Step","Fac Loc Reg One-Step"] #

	method2 = ["FacLoc","One - Step Taylor", "Random","Random Online ","CRAIG",\
	"Full Data Training","Rand-Reg One-Step","Fac Loc Reg One-Step"] 

else:
	method = ["Sup FacLoc with val","Sup FacLoc with trn","One - Step Taylor", "Random","Random Class Prior ","CRAIG",\
	"Full Data Training","Rand-Reg One-Step","Fac Loc Reg One-Step"] #

	method2 = ["Sup FacLoc with val","Sup FacLoc with trn","One - Step Taylor", "Random","CRAIG",\
	"Full Data Training","Rand-Reg One-Step","Fac Loc Reg One-Step"] #

logfile = open(os.path.join(directory, 'combined_'+data_name + '.txt'), 'w')

title = '|Subset Size (%)|'
for i in method:
	title  = title +str(i) +"|"

title2 = '|Subset Size (%)|'
for i in method2:
	title2  = title2 +str(i) +"|"

print(data_name,file=logfile)

for sel in selection:
	print("\nSelect every",sel,file=logfile)
	val_acc = []
	test_acc =[]
	curr_frac=[]
	for frac in range(len(fractions)):

		val =[]
		tst =[]

		file_path = os.path.join(directory,in_dir[frac],str(sel),data_name+'.txt')

		if not os.path.exists(file_path):
			continue

		with open(file_path) as fp:
		   line = fp.readline()

		   while line:
			   if line[0] != '*':
			   		line = fp.readline()
			   		continue

			   acc = [i.strip() for i in line.strip().split("|")]

			   if len(acc) < 2:
			   		line = fp.readline()
			   		continue

			   val.append(acc[-3])
			   tst.append(acc[-2])
			   line = fp.readline()

		if len(val) > 0:
			val_acc.append(val)
			test_acc.append(tst)
			curr_frac.append(fractions[frac])

	if len(val_acc) > 0:
		print(len(val_acc[0]))

		if len(val_acc[0]) == 9:

			print("\nValidation Accuracies",file=logfile)
			print(title,file=logfile)
		
		else:
			print("\nValidation Accuracies",file=logfile)
			print(title2,file=logfile)


		for i in range(len(curr_frac)):
			acc = "|"+str(curr_frac[i])+"|"
			for val in val_acc[i]:
				acc  = acc + val +"|"
			print(acc,file=logfile)


		if len(val_acc[0]) == 9:

			print("\nTest Accuracies",file=logfile)
			print(title,file=logfile)

		else:
			print("\nTest Accuracies",file=logfile)
			print(title2,file=logfile)
		

		for i in range(len(curr_frac)):
			acc = "|"+str(curr_frac[i])+"|"
			for tst in test_acc[i]:
				acc  = acc + tst +"|"
			print(acc,file=logfile)

logfile.close()






