import math
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import numpy as np
import copy, sys, time
from getopt import getopt
from numpy.random import default_rng
from pandas import DataFrame

# Old parser and generator
from linA_snapshot.parser import Parser as Parser_old
from linA_snapshot.lina_gen import NumpyGenerator as GenLinA

# new parser and generator
from expressions.parser import Parser as Parser_new
from LinEx.linex import LinExGenerator as GenLinEx

def code_to_function(code):
	# This implementation returns a python function from the numpy code.
	# TODO: Does not yet support BLAS or anything like that.
	# The function is initially called 'rename_this_function'.
	scope = {'np': np}
	exec(code, scope)  # def rename_this_function(some args): [...]

	def internalFunction(argDict):
		from keyword import iskeyword
		argDict = {(k + '_'if iskeyword(k) else k): v for k, v in argDict.items()}
		start = time.time() # process_time()?
		result = scope['rename_this_function'](**argDict)
		end = time.time() # process_time()?
		return result, end - start
	return internalFunction

# TODO: This function doesn't yet respect all the properties ... PD in particular. Could create them with ATA?
def makeTestData(tree: 'ExpressionTree', seed: int = 0, lowerBound=-1, upperBound=1):
	treeNodes = [tree]
	data = {}

	rng = default_rng(seed)

	while len(treeNodes) > 0:
		node = treeNodes.pop()
		# Make data for variables we do not have yet:
		if node.name.startswith('Var_') and not node.isNumeric():
			vn = node.varName()
			if not vn in data:
				if node.isScalar():
					data[vn] =  rng.uniform(lowerBound, upperBound)
				elif 'diagonal' in node.attributes:
					tmp = rng.uniform(lowerBound, upperBound, size=(node.numericSize()[0]))
					data[vn] = np.diag(tmp)
				# todo: pd before symmetric. Just add a little bit to the diagonal?
				elif 'symmetric' in node.attributes:
					tmp = rng.uniform(lowerBound, upperBound, size=node.numericSize())
					data[vn] = (tmp + tmp.T) / 2 # should also be PSD on the side.
				elif 'triagonal_upper'  in node.attributes or 'triagonal_lower' in node.attributes:
					# Assume that it is 'lower':
					tmp = rng.uniform(lowerBound, upperBound, size=node.numericSize())
					h, w = node.numericSize()
					assert h == w
					for i in range(h):
						tmp[i][i+1:w] = 0
					data[vn] = tmp
					# And flip it if not:
					if 'triagonal_upper' in node.attributes:
						data[vn] = data[vn].T
				else:
					data[vn] = rng.uniform(lowerBound, upperBound, size=node.numericSize())
		# Keep searching for more variables otherwise:
		else:
			treeNodes += node.children()

	return data

def getDeviation(res1, res2, deviation='averageABS'):
	if deviation == 'averageABS' or deviation == 'default':
		return np.sum(np.abs(res1 - res2))
	else:
		print(f"Unknown deviation: '{deviation}'. Using default.")
		return getDeviation(res1, res2, 'default')

def timeSingle(input, which='linex', howOften=20, rngSeed=1337):
	timings = [float('inf')]*howOften

	expTree = Parser_new(input).parse()

	# Make function:
	if which in ['linex', 'linEx', 'LinEx']:
		# TODO: Time code gen! (Separate function; only pass result after all?
		try:
			function = code_to_function(GenLinEx(expTree).generate(docString=False, timeout=timeout))
		except Exception as e:
			print("Error in Code generation: ", e)
			return timings

	elif which in ['lina', 'linA', 'LinA']:
		oldTree = Parser_old(input).parse()
		function = code_to_function(GenLinA(oldTree).generate(docString=False))
	else:
		raise ValueError(f"WARNING, unknown generator: {which}!")

	for i in range(howOften):
		data = makeTestData(expTree, rngSeed)
		try:
			_, time = function(data)
			timings[i] = time
		except Exception as e:
			print(f"Error in Code execution #{i}: ", e)
			return timings


	return timings

def timingFull(inp, compare, filePath, writeCSV=True, which="LinEx", howOften=20, rngSeed=1337, timeout=30):
	# Prepare to collect the data:
	exeTimeColumns = ["Exe Time "+str(n) for n in range(howOften)]
	if writeCSV:
		df = DataFrame(columns=["Generator", "Input"]+exeTimeColumns + ["Average", "StdDev", "Min", "Max"])

	# Execute a single example, if given.
	total = 0
	if inp:
		print(f"\n[Timing {total}]")
		timings = timeSingle(inp, which, howOften, rngSeed)

		average = sum(timings)/howOften
		stdev = math.sqrt(sum((x-average)**2 for x in timings)/howOften)
		minTime = min(timings)
		maxTime = max(timings)

		print("\tAvg", average, "StDev", stdev, "min", minTime, "max:", maxTime, sep="\t")
		if writeCSV:
			df.loc[total] = [which, inp] + timings + [average, stdev, minTime, maxTime]

		total += 1

	# Execute a file of examples, if given:
	if filePath:
		with open(filePath) as file: #matrixcalculus.examples or basic.examples
			for nr, line in enumerate(file):
				# Skip comments or emtpy lines:
				stripped_line = line.strip()
				if stripped_line.startswith("#") or not stripped_line:
					continue

				# Get the input for LinEx and LinA:
				if compare:
					if "||" in line:
						parts = stripped_line.split("||", maxsplit=1)
						lineEx = parts[0]
						lineA = parts [1]
					else:
						lineEx = stripped_line
						lineA = stripped_line
				else:
					lineEx = stripped_line
					lineA = ""

				print(f"\n[Timing {total} | line {nr+1}]")
				print(lineEx)

				timings = timeSingle(input=lineEx, which=which, howOften=howOften, rngSeed=rngSeed)

				average = sum(timings) / howOften
				stdev = math.sqrt(sum((x - average) ** 2 for x in timings) / howOften)
				minTime = min(timings)
				maxTime = max(timings)
				# TODO: Filter inf results (that stem from execution errors)

				print("\tAvg", average, "StDev", stdev, "min", minTime, "max:", maxTime, sep="\t")
				if writeCSV:
					df.loc[total] = [which, lineEx] + timings + [average, stdev, minTime, maxTime]

				total += 1

		if writeCSV:
			if isinstance(writeCSV, str):
				path = writeCSV
			else:
				path = "tmp_exe.csv"
			df.to_csv(path)

def testSingle(inp="5*A*matrix(1)*B*A*C+F", testSeed: int=1337, testExecution=True, compareLinA="5*A*matrix(1)*B*A*C+F",
			   cse=True, opt=False, oriented=True, verbose=True):
	"""
	This is mainly to check whether LinEx works and matches LinA in terms of results. This is not ideal for timing tests.
	"""

	genTimeLinEx = False
	genTimeLinA = False
	exeTimeLinEx = False
	exeTimeLinA = False
	deviation = float('inf')

	# Generate code for LinEx
	try:
		if verbose: print("Generating Code (LinEx) ...")
		genStart = time.time() # process_time()?
		expTree_new = Parser_new(inp).parse()
		ng = GenLinEx(expTree_new.copy(), cse=cse, orientedVectors=oriented)
		linex_code = ng.generate(verbosity=1, timeout=timeout)
		genTimeLinEx = time.time() - genStart # process_time()?
		if verbose: print(f"Done after {genTimeLinEx}s.")
	except Exception as e:
		print("Aborted: Error during LinEx code generation!")
		print(str(e))
		return genTimeLinEx, genTimeLinA, exeTimeLinEx, exeTimeLinA, deviation

	# Generate code for LinA
	if compareLinA:
		try:
			if verbose: print("Generating Code (LinA) ...")
			genStart = time.time() # process_time()?
			expTree_old = Parser_old(compareLinA).parse()
			ng = GenLinA(expTree_old.copy(), cse=cse, optimize=opt, orientedVectors=oriented)
			lina_code = ng.generate()
			genTimeLinA = time.time() - genStart # process_time()?
			if verbose: print(f"Done after {genTimeLinA}s.")
		except Exception as e:
			print("Aborted: Error during LinA code generation!")
			print(str(e))
			return genTimeLinEx, genTimeLinA, exeTimeLinEx, exeTimeLinA, deviation

	if testExecution:
		# Create function for LinEx:
		try:
			if verbose: print("Generating Function (LinEx) ...")
			linex_fun = code_to_function(linex_code)
			if verbose: print("Done.")
		except Exception as e:
			print("Aborted: Error during LinEx function parsing!")
			print(str(e))
			return genTimeLinEx, genTimeLinA, exeTimeLinEx, exeTimeLinA, deviation

		# Create function for LinA:
		if compareLinA:
			try:
				if verbose: print("Generating Function (LinA) ...")
				lina_fun = code_to_function(lina_code)
				if verbose: print(f"Done.")
			except Exception as e:
				print("Aborted: Error during LinA function parsing!")
				print(str(e))
				return genTimeLinEx, genTimeLinA, exeTimeLinEx, exeTimeLinA, deviation

		# Make sample data (requires the NEW expTree):
		if verbose: print("Generating random data ... ")
		argDict = makeTestData(expTree_new, testSeed)
		if verbose: print("Done.")

		# Execute LinEx:
		try:
			if verbose: print("Executing LinEx")
			linex_res, exeTimeLinEx = linex_fun(argDict)
			if verbose: print(f"Done after {exeTimeLinEx}s.")
		except Exception as e:
			print("Aborted: There was an error executing the LinEx function.")
			print(str(e))
			return genTimeLinEx, genTimeLinA, exeTimeLinEx, exeTimeLinA, deviation

		# Execute LinA and compare:
		if compareLinA:
			try:
				if verbose: print("Executing LinA")
				lina_res, exeTimeLinA = lina_fun(argDict)
				if verbose: print(f"Done after {exeTimeLinA}s.")
			except Exception as e:
				print("Aborted: There was an error executing the LinA function.")
				print(str(e))
				return genTimeLinEx, genTimeLinA, exeTimeLinEx, exeTimeLinA, deviation

			if verbose: print("Comparing ...")
			deviation = getDeviation(linex_res, lina_res)
			if verbose: print(f"Done (Deviation: {deviation})")

	return genTimeLinEx, genTimeLinA, exeTimeLinEx, exeTimeLinA, deviation

def testFull(inp, compare, filePath, execute, writeCSV, seed, timeout):
	# Prepare to collect the data:
	total, successes, fasterCount, nancount = 0, 0, 0, 0
	if writeCSV:
		df = DataFrame(columns=["LinEx input", "LinEx generation (s)", "LinEx execution (s)", "LinA input", "LinA generation (s)", "LinA execution (s)", "Result Deviation"])

	# Execute a single example, if given.
	if inp:
		print(f"\n[Test {total}]")
		res = testSingle(inp=inp, testSeed=seed, testExecution=execute, compareLinA=compare)
		genTimeLinEx, genTimeLinA, exeTimeLinEx, exeTimeLinA, deviation = res
		successes += 1 if deviation and deviation < 10e-8 else 0
		fasterCount += 1 if exeTimeLinEx < exeTimeLinA else 0
		nancount += 1 if exeTimeLinA is False else 0

		if writeCSV:
			df.loc[total] = inp, str(genTimeLinEx), str(exeTimeLinEx), str(compare), str(genTimeLinA), str(exeTimeLinA), str(deviation)

		total += 1

	# Execute a file of examples, if given:
	if filePath:
		with open(filePath) as file: #matrixcalculus.examples or basic.examples
			for nr, line in enumerate(file):
				# Skip comments or emtpy lines:
				stripped_line = line.strip()
				if stripped_line.startswith("#") or not stripped_line:
					continue

				# Get the input for LinEx and LinA:
				if compare:
					if "||" in line:
						parts = stripped_line.split("||", maxsplit=1)
						lineEx = parts[0]
						lineA = parts [1]
					else:
						lineEx = stripped_line
						lineA = stripped_line
				else:
					lineEx = stripped_line
					lineA = ""

				print(f"\n[Test {total} | line {nr+1}]")
				print(lineEx)
				res = testSingle(inp=lineEx, testSeed=seed, testExecution=execute, compareLinA=lineA)
				genTimeLinEx, genTimeLinA, exeTimeLinEx, exeTimeLinA, deviation = res
				successes += 1 if deviation < 10e-8 else 0
				fasterCount += 1 if exeTimeLinEx < exeTimeLinA else 0
				nancount += 1 if exeTimeLinA is False else 0

				if writeCSV:
					df.loc[total] = lineEx, str(genTimeLinEx), str(exeTimeLinEx), lineA, str(genTimeLinA), str(exeTimeLinA), str(deviation)

				total += 1

		if compare:
			if total == successes:
				print(f"\n[SUCCESS] Successfully completed {total} tests. Faster despite caching in {fasterCount}. {nancount} impossible to calculate.")
			else:
				print(f"\n[FAILURE] Successfully completed only {successes} of {total} tests. Faster despite caching in {fasterCount}. {nancount} impossible to calculate.")

		if writeCSV:
			if isinstance(writeCSV, str):
				path = writeCSV
			else:
				path = "tmp_gen.csv"
			df.to_csv(path)

def parseCommandArgs():
	opts, _ = getopt(sys.argv[1:], "m:i:c:f:e:w:s:t:")
	# Start with defaults:
	mode = "generate"
	inp = ""
	compare = "same"
	filePath = None
	execute = True
	writeCSV = False
	seed = 0
	timeout = 30 # 30 minutes

	for name, value in opts:
		if '-m' == name:
			mode = value
		elif '-i' == name:
			inp = value
		elif '-c' == name:
			compare = value
		elif '-f' == name:
			filePath = value
		elif '-e' == name:
			execute = bool(value)
		elif '-w' == name:
			writeCSV = bool(value)
		elif '-s' == name:
			seed = int(value)
		elif '-t' == name:
			timeout = float(value)

	if compare == "same":
		compare = inp
	return mode, inp, compare, filePath, execute, writeCSV, seed, timeout

if __name__ == "__main__":
	mode, inp, compare, filePath, execute, writeCSV, seed, timeout = parseCommandArgs()

	if timeout:
		print(f"The timelimit for all LinEx generations is {timeout} minutes.")

	if mode in ['g', 'gen', 'generate']:
		testFull(inp, compare, filePath, execute, writeCSV, seed, timeout)
	elif mode in ['t', 'time', 'e', 'exe', 'execute']:
		timingFull(inp, compare, filePath, writeCSV, rngSeed=seed, timeout=timeout)
	else:
		raise ValueError("Unknown mode: "+mode)


