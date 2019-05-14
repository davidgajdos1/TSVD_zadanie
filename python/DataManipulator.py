from pyspark import SparkContext

def splitDatafile(fileName, amount = 1000):
	f = open(fileName);
	f2 = open(fileName.replace(".csv", "_split.csv"), "w+");
	
	lines = [];
	for i in range(0, amount):
		lines.append(f.readline());
	f.close();
	
	for line in lines:
		f2.write(line);
	f2.close();

class DataManipulator(SparkContext):
	def __init__(self, **kwargs):
		super(DataManipulator, self).__init__(**kwargs);
		
	def loadCSV(self, fileLocation, delimiter = ",", paired = False):
		csv = self.textFile(fileLocation).map(lambda line: line.split(delimiter));
		return csv if not paired else csv.map(lambda x: (x[0], x[1:]));
	
	def joinCSV(self, RDDlist, header = False):
		if type(RDDlist) != list or len(RDDlist) == 0: return None;
		
		joined = RDDlist[0];
		index = joined.first()[0] if header else None;
		
		# join RDDs
		for rdd in RDDlist[1:]:
			joined = joined.join(rdd);
			joined = joined.mapValues(lambda x: x[0] + x[1]);
		
		# find and subtract header
		if header:
			_header = joined.filter(lambda x: x[0] == index);
			joined = joined.subtractByKey(_header);
		
		return joined, _header.collect()[0] if header else None;
	
	def split(self, set, ratio, seed = 1337):
		return set.randomSplit([ratio, 1.0-ratio], seed)