from expressions import Parser
from LinEx.linex import LinExGenerator
from expressions.dimension import Dimension
from testing.testBench import makeTestData, code_to_function

showNumbers = False # Set to true if you're ready to fill your console window ...

if __name__ == "__main__":
	example = "A*B*C+A*B*C"
	Dimension.defaultNumeric = 42 # It usually uses 100 as the default for unspecified dimensions.

	p = Parser(example)
	et = p.parse()

	print("Generating Code with LinEx (timeout 5 minutes).")
	lg = LinExGenerator(et)
	code = lg.generate(timeout=5, verbosity=2)
	print("\n", code, "\n")

	print("Constructing function object and test data")
	function = code_to_function(code)
	
	testData = makeTestData(et, seed=42)
	if showNumbers:
		for key, val in testData.items():
			print(key, val, sep="\n")
	else:
		for key, val in testData.items():
			print(key, val.shape, sep="\t")
			
	print("Executing the function.")
	res, time = function(testData)
	print(f"Done after {time}s.")
	
	if showNumbers:
		print(res)
	else:
		print("result", res.shape)
