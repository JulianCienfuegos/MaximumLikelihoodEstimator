"""
This program will give an optimal (maybe) parameterization for Dirichlet 
distributions. Parameters are elicited from experts, but the numbers don't 
often match - so we get a best fit given the variety of opinions provided
by experts.

A question is read in from a file. 
Then numbers are read in from the RandomNumberTable, one row at a time. 
For each row, a row in the gamma table is generated. (This row does not need to be stored after it is normalized)
Then a row in the normalized gamma table is generated.

To use this function you must have a csv file in your current directory with the elicited parameters from experts. 
You also must have a csv file called rand Table, which contains the randomly generated numbers to be used in later calculations.
To run, type main.py and then fill in the boxes on the GUI!
The resultant parametrizations are going to be in the csv file you specified.


"""

from question import *
from Tkinter import *

#GUI stuff
master = Tk()
Label(master, text="Input file").grid(row=0)
Label(master, text="Output file").grid(row=1)
e1 = Entry(master)
e2 = Entry(master)
e1.insert(10,"ElicitedParameters.csv")
e2.insert(10,"BestParameterFit.csv")
e1.grid(row=0, column=1)
e2.grid(row=1, column=1)
def getFiles(event=None):
	global InFile, OutFile
	InFile = e1.get()
	OutFile = e2.get()
	master.destroy()
master.bind('<Return>', getFiles)
Button(master, text='Calculate', command=getFiles).grid(row=3, column=0, sticky=W, pady=4)
mainloop( )

# Get the random number table.
with open('randTable.csv', 'rb') as data:
	rows = csv.reader(data)
	randTable = [[float(item) for number, item in enumerate(row)] for row in rows]	

# Get the questions
with open(InFile, 'rb') as data:
		rows = csv.reader(data)
		Q = [r[0] for r in rows if r[0] != '']

# Calculate the parametrizations. 
output = []
for q in Q:
	a = Question(q, InFile, randTable)
	fun, params = a.maxLogLikelihood()
	params.insert(0, fun)
	output.append(params)
	
# Write the parameterizations to a file. 
# Each line in the file gives : the maxloglikelihood, followed by the n parameters giving that extreme value subject to some optimizing function.
with open(OutFile, 'wb') as f:
	writer = csv.writer(f)
	for line in output:	
		writer.writerow(line)