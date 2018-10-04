from pyspark.sql import SparkSession
from pyspark import SparkConf
from pyspark.sql.types import *
from pyspark.ml.feature import Imputer, RFormula
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, CrossValidatorModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import col, ceil

conf = SparkConf().setMaster('local').setAppName('My app')
spark = SparkSession\
		.builder\
		.appName('Prediction using pipeline')\
		.config('spark.some.config.option', 'some-value', conf)\
		.getOrCreate()

trainingFilePart = 'train.csv'
customSchema = StructType([StructField('PassengerId', IntegerType(),False),
						   StructField('Survived', IntegerType(), False),
						   StructField('PClass', StringType(), True),
						   StructField('Name', StringType(), False),
						   StructField('Sex', StringType(), True),
						   StructField('Age', FloatType(), True),
						   StructField('SibSb', StringType(), True),
						   StructField('Parch', StringType(), True),
						   StructField('Ticket', StringType(), True),
						   StructField('Fare', FloatType(), True),
						   StructField('Cabin', StringType(), True),
						   StructField('Embarked', StringType(), True)])
rawTraining = spark.read.csv(trainingFilePart, header = True, schema = customSchema)
selectedTraining = rawTraining.select(col('Survived').alias('label'), 'PClass', 'Sex', 'Age', 'Fare')
addingColTraining = selectedTraining.withColumn('Missing_Age', selectedTraining['Age'].isNull()).withColumn('Missing_Fare', selectedTraining['Fare'].isNull())

'''build pipeline'''
imputer = Imputer(inputCols = ['Age', 'Fare'], outputCols = ['Out_Age', 'Out_Fare'])
rformula = RFormula(formula = '~ Sex + Out_Age + Missing_Age + Out_Fare + Missing_Fare', featuresCol = 'features')
lr = LogisticRegression(family = 'binomial')
pipeline = Pipeline(stages = [imputer, rformula, lr])

'''build validation'''
evaluator = BinaryClassificationEvaluator()
grid = ParamGridBuilder().addGrid(lr.maxIter, [10, 50, 100])\
						 .addGrid(lr.regParam, [0.0, 0.01, 0.03, 0.1, 0.3])\
						 .addGrid(lr.elasticNetParam, [0.0, 0.01, 0.03])\
						 .build()
cv = CrossValidator(estimator = pipeline, estimatorParamMaps = grid, evaluator = evaluator, numFolds = 5)

model = cv.fit(addingColTraining)
bestFitness = max(model.avgMetrics)
print('best fitness = ', bestFitness)
bestModel = model.bestModel
#bestModel.save('trainning_model_version3')
#model = PipelineModel.load('trainning_model2')
print("type model = ",bestModel)
print(bestModel.stages[2].explainParam('maxIter'))
print(bestModel.stages[2].explainParam('regParam'))
print(bestModel.stages[2].explainParam('elasticNetParam'))
filePath = 'test.csv'
customSchema = StructType([StructField('PassengerId', IntegerType(), False),
						   StructField('PClass', StringType(), True),
						   StructField('Name', StringType(), False),
						   StructField('Sex', StringType(), True),
						   StructField('Age', FloatType(), True),
						   StructField('SibSb', StringType(), True),
						   StructField('Parch', StringType(), True),
						   StructField('Ticket', StringType(), True),
						   StructField('Fare', FloatType(), True),
						   StructField('Cabin', StringType(), True),
						   StructField('Embarked', StringType(), True)])
rawTesting = spark.read.csv(filePath, header = True, schema = customSchema)
selectedTesting = rawTesting.select('PassengerId', 'PClass', 'Sex', 'Age', 'Fare')
addingColTesting = selectedTesting.withColumn('Missing_Age', selectedTesting['Age'].isNull()).withColumn('Missing_Fare', selectedTesting['Fare'].isNull())

result = model.transform(addingColTesting).select('PassengerId', ceil(col('prediction')).alias('Survived'))
result.write.csv('output_version3.csv', header = True, mode = 'overwrite')