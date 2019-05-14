from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.ml import Pipeline
from pyspark.ml.classification import LinearSVC, DecisionTreeClassifier, DecisionTreeClassificationModel, RandomForestClassifier, NaiveBayes, NaiveBayesModel
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer, VectorAssembler
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.clustering import KMeans
from pyspark.sql.functions import  *
import pandas as pd
import numpy as np

#nacitanie CSV pomocou sparku do dataframe-ov
def loadCSV():
    data_frame1 = spark.read.format("csv").option("inferSchema","true").option("header","true").load("../data/Accidents_split.csv")
    data_frame2 = spark.read.format("csv").option("inferSchema","true").option("header","true").load("../data/Casualties_split.csv")
    data_frame3 = spark.read.format("csv").option("inferSchema","true").option("header","true").load("../data/Vehicles_split.csv")

    #vymazanie NULL-\^.^/-ovych hodnot
    data_frame1 = data_frame1.dropna()
    data_frame2 = data_frame2.dropna()
    data_frame3 = data_frame3.dropna()

    return data_frame1, data_frame2, data_frame3

#spojenie viacero dataframe-ov do jedneho dataframe-u a spracovanie chybajucich hodnot
def joinCSVs(data_frame1, data_frame2, data_frame3):
    temp_data_frame = data_frame1.join(data_frame2, "Accident_Index")
    final_data_frame = temp_data_frame.join(data_frame3, "Accident_Index")  
    
    return final_data_frame


#transformacia cieloveho atributu, 1 ak bola smrtelna nehoda, 0 ak nie
def transformTargetAttribute(data_frame):
    return data_frame.withColumn('Casualty_Severity_Index', when(data_frame.Casualty_Severity==3, 0).otherwise(1))

#vytvorenie zmensenej vzorky z povodneho datasetu
def sampling(data_frame, ratio, seed):
    #data_frame = data_frame.sample(False, ratio, seed)
    fractions = {1: 0.9, 0: 0.1}
    data_frame = data_frame.sampleBy("Casualty_Severity_Index", fractions , seed)
    return data_frame
    
    
#rozdelenie dat na trenovaciu a testovaciu mnozinu v pomere
def splitTrainTest(data_frame, train_ratio):
    train_data, test_data = data_frame.randomSplit([train_ratio, 1-train_ratio])
    return train_data, test_data

#konvertovanie nominalnych hodnot na numericke  
def nominalToNumeric(data_frame):
    table1 = StringIndexer(inputCol = 'Accident_Index', outputCol = 'Id_num')
    table2 = StringIndexer(inputCol = 'Local_Authority_(Highway)', outputCol = 'LAH_num')
    table3 = StringIndexer(inputCol = 'LSOA_of_Accident_Location', outputCol = 'LoAL_num')
    
    _table1 = table1.fit(data_frame).transform(data_frame)
    _table2 = table2.fit(_table1).transform(_table1)
    _table3 = table3.fit(_table2).transform(_table2)
    _saveTable3 = _table3
    
    _table3 = _table3.drop('Accident_Index')
    _table3 = _table3.drop('LSOA_of_Accident_Location')
    _table3 = _table3.drop('Local_Authority_(Highway)')
    
    return _saveTable3, _table3

#konvertovanie numerickych hodnot na nominalne
def numericToNominal(data_frame):
    converter = IndexToString(inputCol='LoAL_num', outputCol='originalCategory')
    converted = converter.transform(data_frame)

#vytvorenie histogramu z data_frameu a prislusneho indexu    
def histogram(data_frame, index):
    return data_frame.groupBy(index).count().show()

#zobrazenie statistik numerickych hodnot
def statisticsOfNumericValues(numeric_data, index):
    statistics = numeric_data.describe([index])
    statistics.show()
    
def inputDataForModel(data_frame):
    vector_data = VectorAssembler(inputCols=["Location_Easting_OSGR", "Location_Northing_OSGR", "Longitude", "Latitude", "Police_Force",
                                          "Number_of_Vehicles", "Number_of_Casualties", "Day_of_Week", "Local_Authority_(District)"
                                        , "1st_Road_Class", "1st_Road_Number", "Road_Type", "Speed_limit", "Junction_Detail",
                                         "Junction_Control", "2nd_Road_Class", "2nd_Road_Number", "Pedestrian_Crossing-Human_Control",
                                         "Pedestrian_Crossing-Physical_Facilities", "Light_Conditions", "Weather_Conditions", "Road_Surface_Conditions",
                                         "Special_Conditions_at_Site", "Carriageway_Hazards", "Urban_or_Rural_Area", "Did_Police_Officer_Attend_Scene_of_Accident",
                                           "Casualty_Reference", "Casualty_Class", "Sex_of_Casualty",
                                         "Age_of_Casualty", "Age_Band_of_Casualty", "Accident_Severity", "Pedestrian_Location", "Pedestrian_Movement",
                                         "Car_Passenger", "Bus_or_Coach_Passenger", "Pedestrian_Road_Maintenance_Worker", "Casualty_Type", "Casualty_Home_Area_Type",
                                         "Vehicle_Type", "Towing_and_Articulation", "Vehicle_Manoeuvre", "Vehicle_Location-Restricted_Lane", "Junction_Location",
                                         "Skidding_and_Overturning", "Hit_Object_in_Carriageway", "Vehicle_Leaving_Carriageway", "Hit_Object_off_Carriageway",
                                         "1st_Point_of_Impact", "Was_Vehicle_Left_Hand_Drive?", "Journey_Purpose_of_Driver", "Sex_of_Driver", "Age_of_Driver",
                                         "Age_Band_of_Driver", "Engine_Capacity_(CC)", "Propulsion_Code", "Age_of_Vehicle", "Driver_IMD_Decile",
                                         "Driver_Home_Area_Type"], 
                                         outputCol="features").transform(data_frame)

    return vector_data

def _KMeans(training_data):
    kmeans = KMeans(featuresCol="features", k=2, seed=1234) #k = zhluky
    kmeans_model = kmeans.fit(training_data)
    clusters = kmeans_model.transform(training_data)
    
    print(kmeans_model.clusterCenters()[0])

def decisionTree(training_data, test_data):
    tree_classifier = DecisionTreeClassifier(
            featuresCol="features",                         # datovy stlpec obsahujuci vektor vstupnych atributov
            labelCol="Casualty_Severity_Index",             # datovy stlpec obsahujuci cielovy atribut (indexy tried)
            impurity = "entropy",                           # pre vyber atributov pri deleni sa pouzije kriterium informacneho zisku
            maxDepth=5)                                     # ohranicime maximalnu hlbku generovaneho stromu

    tree_model = tree_classifier.fit(training_data)
    predictions = tree_model.transform(test_data)

    test_error = predictions.filter(predictions["prediction"] != predictions["Casualty_Severity_Index"]).count() / float(test_data.count())
    print "Testing error: {0:.4f}".format(test_error)
    return predictions

def randomForrest(train_data, test_data):
    # Train a RandomForest model.
    rf = RandomForestClassifier(featuresCol="features", labelCol="Casualty_Severity_Index")

    rf_model = rf.fit(train_data)
    predictions = rf_model.transform(test_data)

    test_error = predictions.filter(predictions["prediction"] != predictions["Casualty_Severity_Index"]).count() / float(test_data.count())
    print "Testing error: {0:.4f}".format(test_error)
    return predictions

def linearSVM(training_data, test_data):
    svm_classifier = LinearSVC(featuresCol="features", labelCol="Casualty_Severity_Index")

    svmModel = svm_classifier.fit(training_data)
    predictions = svmModel.transform(test_data)

    test_error = predictions.filter(predictions["prediction"] != predictions["Casualty_Severity_Index"]).count() / float(test_data.count())
    print "Testing error: {0:.4f}".format(test_error)
    return predictions

def naiveBayes(training_data, test_data):
    naivebayes = NaiveBayes(featuresCol="features", labelCol="Casualty_Severity_Index")
    model = naivebayes.fit(training_data)
    prediction = model.transform(test_data)
    return prediction

def evaluation(predictions):
    tp = float(predictions.filter((predictions["prediction"] == 1.0) & (predictions["Casualty_Severity_Index"] == 1.0)).count())
    tn = float(predictions.filter((predictions["prediction"] == 0.0) & (predictions["Casualty_Severity_Index"] == 0.0)).count())
    fp = float(predictions.filter((predictions["prediction"] == 1.0) & (predictions["Casualty_Severity_Index"] == 0.0)).count())
    fn = float(predictions.filter((predictions["prediction"] == 0.0) & (predictions["Casualty_Severity_Index"] == 1.0)).count())

    #print("True Positive " + str(tp))
    #print("True Negative " + str(tn))
    #print("False Positive " + str(fp))
    #print("False Negative " + str(fn))
    
    #podla vzorcov
    recall =  tp/ (tp + fn)
    accuracy = (tp + tn) / (tp + fn + fn + fp)
    precision = tp/(tp + fp)
    f_one_score = 2*((precision * recall) / (precision + recall))
    
    return recall, accuracy, f_one_score, precision
    
def createEvalTables(decisionTree_prediction, randomForrest_prediction, linearSVM_prediction):        
    recall1, accuracy1, f_one_score1, precision1 = evaluation(decisionTree_prediction)
    recall2, accuracy2, f_one_score2, precision2 = evaluation(randomForrest_prediction)
    recall3, accuracy3, f_one_score3, precision3 = evaluation(linearSVM_prediction)
    
    dat = np.array([["DecisionTree", str(recall1), str(accuracy1), str(f_one_score1), str(precision1)],["RandomForest", str(recall2), accuracy2, str(f_one_score2), str(precision2)],["LinearSVM", str(recall3), str(accuracy3), str(f_one_score3), str(precision3)]],np.str)
    df = pd.DataFrame(dat, columns = ["ModelName","recall", "accuracy", "f_one_score", "precision"])
    #df.show()
    file_name = "../data/Model_Results.csv"
    df.to_csv(file_name, sep=',')
    #display(df)

if __name__ == "__main__":
    print ('------- Program starting -------')
    #vytvorenie spark contextu a spark session
    sc = SparkContext()
    spark = SparkSession(sc)
    
    #nacitanie csv
    print ('------- Loading Csv -------')
    data_frame1, data_frame2, data_frame3 = loadCSV()
    
    #spojeny dataframe
    print ('------- Joining Csvs -------')
    data_frame = joinCSVs(data_frame1, data_frame2, data_frame3)
    
    print('------- Transforming target attribute -------')
    data_frame_transformed = transformTargetAttribute(data_frame)
    #print ('First row of transformed: '+ data_frame.show(1))
    #Nacitanie 1% hodnot z povodneho datasetu
    _data_frame = sampling(data_frame_transformed, 0.5, None)
    print ('Taking this number of rows from dataset: '+ `_data_frame.count()`)
    
    #rozdelenie dat na trenovaciu a testovaciu mnozinu 
    train_data, test_data = splitTrainTest(_data_frame, 0.6)
    
    print ('Train data count: ' + `train_data.count()`)
    print ('Test data count: ' + `test_data.count()`)
    
    print ('------- Converting nominal data to numeric -------')
    #Nominalne hodnoty na numericke
    print ('------- Converting nominal train data to numeric -------')
    saved_nom_train_value, num_train_data = nominalToNumeric(train_data)
    print ('------- Converting nominal test data to numeric -------')
    saved_nom_test_value, num_test_data = nominalToNumeric(test_data)
    
    print ('------- Converting numeric data to nominal -------')
    #Numericke hodnoty na nominalne
    print ('------- Converting numeric train data to nominal -------')
    nom_train_data = numericToNominal(saved_nom_train_value)
    print ('------- Converting numeric test data to nominal -------')
    nom_test_data = numericToNominal(saved_nom_test_value)

    #Statistiky numerickych dat
    print ("\n------- Showing the statistics of certain numeric attribute -------")
    statisticsOfNumericValues(num_train_data, "Longitude")
    
    #Histogram
    print('------- Showing histogram -------')
    histogram(_data_frame, "Casualty_Severity_Index")
    
    inputTrainData = inputDataForModel(num_train_data)
    inputTestData = inputDataForModel(num_test_data)
    
    print('------- K-Means clusters centers-------')
    _KMeans(inputTrainData)
    
    print('\n------- DecisionTree -------')
    decisionTree_prediction = decisionTree(inputTrainData, inputTestData)
    print('------- DecisionTree evaluation-------')
    recall, accuracy, f_one_score, precision = evaluation(decisionTree_prediction)
    
    print ("Recall:" + str(recall))
    print ("Accuracy:" + str(accuracy))
    print ("F1 score:"+ str(f_one_score))
    print ("Precision:" + str(precision))  
    
    print('\n------- RandomForrest -------')
    randomForrest_prediction = randomForrest(inputTrainData, inputTestData)
    print('------- RandomForrest evaluation -------')
    recall, accuracy, f_one_score, precision = evaluation(randomForrest_prediction)
    print ("Recall:" + str(recall))
    print ("Accuracy:" + str(accuracy))
    print ("F1 score:"+ str(f_one_score))
    print ("Precision:" + str(precision))  
    
    print('\n------- LinearSVM -------')
    linearSVM_prediction = linearSVM(inputTrainData, inputTestData)
    print('------- LinearSVM evaluation-------')
    recall, accuracy, f_one_score, precision = evaluation(linearSVM_prediction)
    print ("Recall:" + str(recall))
    print ("Accuracy:" + str(accuracy))
    print ("F1 score:"+ str(f_one_score))
    print ("Precision:" + str(precision)) 
    
    print('\n------- Creating model comparision table -------')
    createEvalTables(decisionTree_prediction, randomForrest_prediction, linearSVM_prediction)