import org.apache.spark.ml.classification.{LogisticRegression, NaiveBayes}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer, VectorAssembler}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.mllib.evaluation.RegressionMetrics
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
//    val attrDF = ss.read.option("header", "true").csv("attributes.csv")
//
    val trainDescDF = trainDF.join(descDF, "product_uid")

    trainDF.repartition(4)
    val DF_0 = trainDescDF
      .withColumn("relevance", udf_toDouble($"relevance"))
    //
    //    // Create tf-idf features
    val tokenizer = new Tokenizer()
    val hashingTF = new HashingTF()
    val idf = new IDF()
    //
    ////    ----------------------------------------Ayto thelei optimize....---------------------------------
    tokenizer.setInputCol("product_title").setOutputCol("tok_title")
    hashingTF.setInputCol("tok_title").setOutputCol("titleFeatures").setNumFeatures(20000)
    idf.setInputCol("titleFeatures").setOutputCol("titleTFIDF")
//
    val DF_1 = hashingTF.transform(tokenizer.transform(DF_0)).drop("product_title").drop("tok_title")
    val DF_2 = idf.fit(DF_1).transform(DF_1).drop("titleFeatures")

    tokenizer.setInputCol("search_term").setOutputCol("tok_term")
    hashingTF.setInputCol("tok_term").setOutputCol("termFeatures").setNumFeatures(20000)
    idf.setInputCol("termFeatures").setOutputCol("termTFIDF")

    val DF_3 = hashingTF.transform(tokenizer.transform(DF_2)).drop("search_term").drop("tok_term")
    val DF_4 = idf.fit(DF_3).transform(DF_3).drop("termFeatures")


    tokenizer.setInputCol("product_description").setOutputCol("tok_desc")
    hashingTF.setInputCol("tok_desc").setOutputCol("descFeatures").setNumFeatures(20000)
    idf.setInputCol("descFeatures").setOutputCol("descTFIDF")

    val DF_5 = hashingTF.transform(tokenizer.transform(DF_4)).drop("product_description").drop("tok_desc")
    val DF_6 = idf.fit(DF_5).transform(DF_5).drop("descFeatures")

    val assembler = new VectorAssembler()
      .setInputCols(Array( "titleTFIDF", "termTFIDF", "descTFIDF"))
      .setOutputCol("features")


    val finalDF = assembler.transform(DF_6).withColumn("relevance", udf_toDouble($"relevance")).withColumnRenamed("relevance", "label")
      .drop("titleTFIDF")
      .drop("termTFIDF")
      .drop("descTFIDF")
//////    ----------------------------------------------------------------------------------------------------------------------
////
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


    val lr = new LinearRegression()
      .setMaxIter(100)
      .setRegParam(0.1)
      .setElasticNetParam(0)
      .fit(trainingData)

    val trainingSum = lr.summary

    print("Mean Squered Error: ")
    println(trainingSum.meanSquaredError)


    println("Done!")

  }


}
