#!/usr/bin/env python

import re
import sys
import os

allfiles={}

beginfactor=0.1 		# show data from 10% of the iterations upwards (so the auto y-scale works better)
progress_files=["compute_prob", "iter", "compute_objf"]
train_flags=["train", ".tr"]
progress_types=['train', 'valid']

if len(sys.argv)>1:
	exp=str(sys.argv[1])
else:
	exp=""

# scan for xml files
for root, dirs, files in os.walk('./' + exp, followlinks=True):
	for name in files:
		if ( os.path.splitext(name)[1]=='.log' and any(item in os.path.splitext(name)[0] for item in progress_files) ) :	
			# if len(sys.argv)>1:
			#	experiment=exp
			# else:			
			experiment=re.sub('^\.\/exp\/|\/log$', '', root)
			if any(item in os.path.splitext(name)[0] for item in train_flags):
				type='train'
			else:
				type='valid'
			if experiment not in allfiles.keys():
				allfiles[experiment]={}
				allfiles[experiment]['train']=[]
				allfiles[experiment]['valid']=[]
			# print (experiment, type, root, name)			
						
			allfiles[experiment][type].append(os.path.join(root, name))
			# print(os.path.join(root, name))

allresults={}
for experiment in allfiles.keys():
	if experiment not in allresults.keys():
		allresults[experiment]={}
		allresults[experiment]['train']=[]
		allresults[experiment]['valid']=[]
	for ptype in progress_types:
		for filename in allfiles[experiment][ptype]:
			m=re.findall(r"(\d+)(?:\.tr|\.cv)?\.log", filename)
			if m:
				iteration=m[-1] # .group(m.lastindex)
		
			accuracy=0
			f=open(filename)
			for line in f:
				m=re.search(r"accuracy is ([-+]?[0-9]*\.?[0-9]+)", line)
				if m:
					accuracy = m.group(1)
				m=re.search(r" is ([-+]?[0-9]*\.?[0-9]+) \+ ([-+]?[0-9]*\.?[0-9]+) = ([-+]?[0-9]*\.?[0-9]+) per frame", line)
				if m:
					accuracy = m.group(3)
				m=re.search(r" is ([-+]?[0-9]*\.?[0-9]+) \+ ([-+]?[0-9]*\.?[0-9]+) per frame", line)
				# m=re.search(r"SMBR objective function is ([-+]?[0-9]*\.?[0-9]+) per frame", line)
				if m:
					accuracy = m.group(1)
				m=re.search(r" accuracy for 'output' is ([-+]?[0-9]*\.?[0-9]+) per frame", line)
				if m:
					accuracy = m.group(1)
				m=re.search(r"FRAME_ACCURACY .* ([-+]?[0-9]*\.?[0-9]+)", line)
				if m:
					accuracy = m.group(1)
			if float(accuracy) != 0:
				size = len(allresults[experiment][ptype])
				if int(iteration) >= size:
					allresults[experiment][ptype].extend([None]*(int(iteration)-size+1))
				allresults[experiment][ptype][int(iteration)]=accuracy

f=open('local/chart_learning_framework.html')
t=open('chart_learning.html', 'w')

for line in f:
	m=re.search(r"INSERT DATA HERE", line)
	if m:
		# t.write("var data = google.visualization.arrayToDataTable([\n")
		# t.write("['Experiment', 'Iteration', 'Train', 'Valid'],\n")
		t.write("var data = new google.visualization.DataTable();\n")		
		t.write("data.addColumn('string', 'Experiment');\n")
		t.write("data.addColumn({type:'number', label:'Iteration'});\n")	
		t.write("data.addColumn({type:'number', role:'data', label: 'Train'});\n")		
		t.write("data.addColumn({type:'number', role:'data', label: 'Valid'});\n")		
		t.write("data.addRows([\n")
		for experiment in allresults.keys():
			start=int(beginfactor*len(allresults[experiment]['train']))+1
			for valueidx in range(start, len(allresults[experiment]['valid'])):
				l="[ '" + experiment + "',"+ str(valueidx) + "," + allresults[experiment]['train'][valueidx] + "," + allresults[experiment]['valid'][valueidx] + " ], \n"
				t.write(l)
	
		t.write("]);\n")
		t.write("var options = {\n title: 'Train Progress " + experiment + "',\n legend: { position: 'bottom' },\n")
		# t.write(" vAxis: {\n  viewWindow:{\n   max: 0,\n   min: -0.3\n  }\n }\n")
		t.write("};\n")	
	
	else:
		t.write(line)
	
sys.exit()		
