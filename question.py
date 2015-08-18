import csv
from scipy.stats import gamma
from scipy.special import gammaln
from scipy.optimize import  minimize
from numpy import ones, sum, log, array, dot, eye, savetxt

def trunc(d):
	"""
	Truncate a decimal to three decimal places.
	"""
	return d*1000//1/1000
	
def safe_log(x, minval=0.0000000001):
    # have to prevent overflow.
	return log(x.clip(min=minval))	
	
def norm_log(l):
	total = sum(l)
	l = [i/total for i in l]
	return [safe_log(i) for i in l]
		
class Question:
	"""
	Question objects know their number, in what file to look for their parameters, and where the randomNumber table is.
	The randTable mst have at least 1000 rows, and the number of rows must be a multiple of 10.
	"""
	def __init__(self, name, f, randTable):
		"""
		The data is gathered from the csv file specified by the 'number'.
		The data is then stored in a list of lists.
		"""
		self.elicited = []
		self.getElicited(name, f)
		self.gammaTable = []
		self.genGammaTable(randTable)
	
	def getElicited(self, name, f):
		"""
		Find the lines in f which correspond to the given question name.
		"""
		with open(f, 'rb') as data:
			rows = list(csv.reader(data))
			startrow = None
			endrow = None
			for r in rows: # find the start rows
				if startrow != None and endrow == None and r[0] != '':
					endrow = rows.index(r)
				if r[0] == name:
					startrow = rows.index(r)
			myrows = rows[startrow:endrow]
			myrows = [r[1:] for r in myrows] # remove the leading number or ''
			# remove blanks from each of myrows, and convert every element to a float.
			for i in range(len(myrows)):
				myrows[i] = [float(j) for j in myrows[i] if j != '']
			self.elicited = [list(i) for i in zip(*myrows)]
			
	def genGammaTable(self, randTable):
		"""
		This function will generate the gamma table, given that it knows about
		elicited parameters and the randomTable.
		The following line of code:
			myRows = gammaRows * trunc(R[0])
		decides how many rows to allot to a given expert's opinion given the weight
		assigned to that expert's parametrization.
		"""
		gammaRows = len(randTable)
		numExperts = len(self.elicited)
		numParams = len(self.elicited[0])
		randRow = 0
		# First, normalize the weights.
		total_w = sum([R[0] for R in self.elicited])
		for R in self.elicited:
			myRows = gammaRows * trunc(R[0]/total_w) 	
			for r in range(int(myRows)):
					l = []
					for n in range(1, numParams):
						prob = randTable[randRow][n-1]
						alpha = R[n]
						l.append(gamma.ppf(prob, alpha))
					l = norm_log(l)
					self.gammaTable.append(l)
					randRow += 1	
		self.gammaTable = array(self.gammaTable)
	
	def maxLogLikelihood(self):
		"""
		This function calculates the maximum logLikelihood and the corresponding optimal
		Dirichelet parametrization.
		"""
		n = len(self.elicited[0]) - 1
		bds = [(0.001, 100) for i in range(n)]
		A = self.gammaTable
		def optimFunc(guess):
			"""
			The python optimization routines give minima. Therefore, we optimize 
			the negative function (note 'fun' has a negative sign). Then we adjust
			the sign of the result to positive after the minimization.
			This function is nested in the maxLogLikelihood function because we need
			access to the matrix A, and this is most easily accomplished with nesting/.
			"""
			p1 = 1000*gammaln(sum(guess))
			p2 = -1000*sum(gammaln(guess))
			vec = guess - ones(len(guess))
			b = dot(A, vec)
			p3 = sum(b)
			fun = -(p1 + p2 + p3)
			return fun
		return -1.0*minimize(optimFunc, ones(n), bounds=bds).fun, list(minimize(optimFunc, ones(n), bounds=bds).x)