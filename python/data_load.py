# TODO: zachovanie rozlozenia cieloveho atributu pri sample?

from DataManipulator import DataManipulator;

dm = DataManipulator(appName="test");
accidents = dm.loadCSV("../data/Accidents_split.csv", paired = True);
vehicles = dm.loadCSV("../data/Vehicles_split.csv", paired = True);
casualties = dm.loadCSV("../data/Casualties_split.csv", paired = True);

joined, header = dm.joinCSV([accidents, vehicles, casualties], header = True);
print(header[1]);

train, test = dm.split(joined, 0.6);
print(joined.count());
print(train.count());
print(test.count());