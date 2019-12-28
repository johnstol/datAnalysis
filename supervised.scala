import org.apache.spark.ml.classification.{LogisticRegression, NaiveBayes}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer, VectorAssembler}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object supervised {

  def main(args: Array[String]): Unit = {

    // Create the spark session first
    val ss = SparkSession.builder().master("local").appName("querySuperApp").getOrCreate()
    import ss.implicits._  // For implicit conversions like converting RDDs to DataFrames
    val currentDir = System.getProperty("user.dir")  // get the current directory
//    val trainFile = "./train.csv"
//    val descriptFile = "./product_descriptions.csv"
    //val outputDir = "file://" + currentDir + "/output"

    // Define some helpful User-Defined Functions (UDFs)
    val udf_toDouble = udf[Double, String]( _.toDouble)
//    println("reading from input file: " + trainFile)

    // Read the contents of the csv file in a dataframe
    val trainDF =  ss.read.option("header", "true").csv("train.csv")
    val descDF =  ss.read.option("header", "true").csv("product_descriptions.csv")
    val trainDescDF = trainDF.join(descDF, "product_uid")
    println("Train:")
    trainDF.printSchema()
    println("Desc:")
    descDF.printSchema()
    println("Joined:")
    trainDescDF.printSchema()
//    trainDescDF.take(3).foreach(println)

    //
//    // Set the number of partitions
    trainDescDF.repartition(4)
//
//    // Create tf-idf features
    val tokenizer = new Tokenizer()
    val hashingTF = new HashingTF()
    val idf = new IDF()

//    ----------------------------------------Ayto thelei optimize....---------------------------------
    tokenizer.setInputCol("product_title").setOutputCol("tok_title")
    hashingTF.setInputCol("tok_title").setOutputCol("titleFeatures").setNumFeatures(20000)
    idf.setInputCol("titleFeatures").setOutputCol("titleTF-IDF")
    val DF_0 = trainDescDF
      .withColumn("relevance", udf_toDouble($"relevance"))
//
    val DF_1 = tokenizer.transform(DF_0)
    val DF_2 = hashingTF.transform(DF_1)
    val DF_3 = idf.fit(DF_2).transform(DF_2)


    tokenizer.setInputCol("search_term").setOutputCol("tok_term")
    hashingTF.setInputCol("tok_term").setOutputCol("termFeatures").setNumFeatures(20000)
    idf.setInputCol("termFeatures").setOutputCol("termTF-IDF")
    val DF_4 = tokenizer.transform(DF_3)
    val DF_5 = hashingTF.transform(DF_4)
    val DF_6 = idf.fit(DF_5).transform(DF_5)

    tokenizer.setInputCol("product_description").setOutputCol("tok_desc")
    hashingTF.setInputCol("tok_desc").setOutputCol("descFeatures").setNumFeatures(20000)
    idf.setInputCol("descFeatures").setOutputCol("descTF-IDF")
    val DF_7 = tokenizer.transform(DF_6)
    val DF_8 = hashingTF.transform(DF_7)
    val DF_9 = idf.fit(DF_8).transform(DF_8)

    val complete = DF_9.select($"product_uid", $"titleTF-IDF", $"descTF-IDF", $"termTF-IDF", $"relevance")
//    complete.printSchema()
//    complete.take(2).foreach(println)

    val assembler = new VectorAssembler()
      .setInputCols(Array("titleTF-IDF", "descTF-IDF", "termTF-IDF"))
      .setOutputCol("features")


    val finalDF = assembler.transform(complete).withColumnRenamed("relevance", "label").drop("titleTF-IDF").drop("descTF-IDF").drop("termTF-IDF")
//    ----------------------------------------------------------------------------------------------------------------------
    
    finalDF.printSchema()
    finalDF.take(2).foreach(println)

//
//    /* ======================================================= */
//    /* ================== CLASSIFICATION ===================== */
//    /* ======================================================= */

//    println("BEFORE TRAINING")
//
//    // Split the data into training and test sets (30% held out for testing)
    val Array(trainingData, testData) = finalDF.randomSplit(Array(0.6, 0.4), seed = 1234L)
//
//    /* =============================================================================== */
//    /* Classification example using Naive Bayes Classifier                             */
//    /* =============================================================================== */
//
//    val NBmodel = new NaiveBayes().fit(trainingData)
    val NBmodel = new NaiveBayes().setModelType("multinomial").setSmoothing(3.0).fit(trainingData)
////
    val predictionsNB = NBmodel.transform(testData)
    predictionsNB.printSchema()
//    //predictionsNB.take(100).foreach(println)
//    //predictionsNB.select("label", "prediction").show(100)
    predictionsNB.show(2)
//
//    // Evaluate the model by finding the accuracy
//    val evaluatorNB = new MulticlassClassificationEvaluator()
//      .setLabelCol("label")
//      .setPredictionCol("prediction")
//      .setMetricName("accuracy")
////
//    val accuracyNB = evaluatorNB.evaluate(predictionsNB)
//    println("Accuracy of Naive Bayes: " + accuracyNB)
//
//
//    /* =============================================================================== */
//    /* Classification example using Logistic Regression Classifier                     */
//    /* =============================================================================== */
//
//    val LRmodel = new LogisticRegression()
//      .setMaxIter(10000)
//      .setRegParam(0.1)
//      .setElasticNetParam(0.0)
//      .fit(trainingData)
//
//    val predictionsLR = LRmodel.transform(testData)
//    predictionsLR.printSchema()
//    predictionsLR.show(2)

    print("Done!")
//    val accuracyLR = evaluatorNB.evaluate(predictionsLR)
//    println("Accuracy of Logistic Regression: " + accuracyLR)

  }


}