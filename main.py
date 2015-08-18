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
# --------------------------------------------------------------------------
# GUI stuff
# --------------------------------------------------------------------------

master = Tk()
Label(master, text="Input file").grid(row=0)
Label(master, text="Output file").grid(row=1)
e1 = Entry(master)
e2 = Entry(master)
e1.insert(10,"X.csv")
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

# -------------------------------------------------------------------------
# Useful function
# -------------------------------------------------------------------------

def log_float(L):
		return [log(float(l)) for l in L]

# --------------------------------------------------------------------------
# Input the file of Xs
# --------------------------------------------------------------------------
with open(InFile, 'rb') as data:
		rows = csv.reader(data)
		X = [log_float(r) for r in rows]
		X = array(X)

# ---------------------------------------------------------------------------
# MaxLogLikelihood Calculation
# ---------------------------------------------------------------------------

def maxLogLikelihood(ln_X_matrix):
	n = len(ln_X_matrix) # num rows in X
	k = len(ln_X_matrix[0]) # num cols in X
	bds = [(0.000001, 100) for i in range(k)]
	def optimFunc(alpha_guess):
		term_1 =  n*gammaln(sum(alpha_guess))
		term_2 = -n*sum(gammaln(alpha_guess))
		vec = alpha_guess - ones(k)
		b = dot(ln_X_matrix, vec)
		term_3 = sum(b)
		fun = -(term_1 + term_2 + term_3)
		return fun
	# if you want the max(f(x)), return the -min(-f(x))
	return -1.0*minimize(optimFunc, ones(k), bounds=bds).fun, list(minimize(optimFunc, ones(k), bounds=bds).x)


# Calculate the parametrizations. 
MLE = maxLogLikelihood(X)
print MLE
