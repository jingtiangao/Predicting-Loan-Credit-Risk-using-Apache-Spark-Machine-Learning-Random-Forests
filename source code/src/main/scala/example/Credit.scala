package example


import breeze.linalg._
import breeze.plot._
import org.apache.spark._
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SQLContext}

object Credit {

  case class Credit(
    creditability: Double,
    balance: Double, duration: Double, history: Double, purpose: Double, amount: Double,
    savings: Double, employment: Double, instPercent: Double, sexMarried: Double, guarantors: Double,
    residenceDuration: Double, assets: Double, age: Double, concCredit: Double, apartment: Double,
    credits: Double, occupation: Double, dependents: Double, hasPhone: Double, foreign: Double
  )

  def parseCredit(line: Array[Double]): Credit = {
    Credit(
      line(0),
      line(1) - 1, line(2), line(3), line(4), line(5),
      line(6) - 1, line(7) - 1, line(8), line(9) - 1, line(10) - 1,
      line(11) - 1, line(12) - 1, line(13), line(14) - 1, line(15) - 1,
      line(16) - 1, line(17) - 1, line(18) - 1, line(19) - 1, line(20) - 1
    )
  }

  def parseRDD(rdd: RDD[String]): RDD[Array[Double]] = {
    rdd.map(_.split(",")).map(_.map(_.toDouble))
  }

  def main(args: Array[String]) {

    val conf = new SparkConf().setAppName("SparkDFebay").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._

    val creditDF = parseRDD(sc.textFile("germancredit.csv")).map(parseCredit).toDF().cache()
    creditDF.registerTempTable("credit")
    //creditDF.printSchema
    //creditDF.show
   // sqlContext.sql("SELECT creditability, avg(balance) as avgbalance, avg(amount) as avgamt, avg(duration) as avgdur  FROM credit GROUP BY creditability ").show

   // creditDF.describe("balance").show
  //  creditDF.groupBy("creditability").avg("balance").show

    val featureCols = Array("balance", "duration", "history", "purpose", "amount",
      "savings", "employment", "instPercent", "sexMarried", "guarantors",
      "residenceDuration", "assets", "age", "concCredit", "apartment",
      "credits", "occupation", "dependents", "hasPhone", "foreign")
    val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
    val df2 = assembler.transform(creditDF)
   // df2.show

    val labelIndexer = new StringIndexer().setInputCol("creditability").setOutputCol("label")
    val df3 = labelIndexer.fit(df2).transform(df2)
    //df3.show

    val splitSeed = 5043
    val Array(trainingData, testData) = df3.randomSplit(Array(0.7, 0.3), splitSeed)

    //数据管理：Array[double]
    //随机森林自身纵向比较首先使用10种不同的深度，1,2,3,4,5,6,7,8,9,10比较AUC
    //再使用10-30分支，比较AUC
    //与单纯的使用决策树比较
    var deepAUC:Array[Double] = new Array[Double](10)
    var treeAUC:Array[Double] = new Array[Double](21)
    var td=new DenseMatrix[Double](10,21)
    val i:Int=0
    for(i<- 1 to 10) {
       deepAUC(i-1) = getAUC(i, 20, trainingData, testData)
    }
   /* for(i<- 10 to 30){
       treeAUC(i-10) = getAUC(3, i, trainingData, testData)
    }
    val j:Int=0
    for(i<- 1 to 10) {
      for(j<- 10 to 30){
        td(i-1,j-10) = getAUC(i, j, trainingData, testData)
      }
    }*/
   /* val rm = new RegressionMetrics(
      predictions.select("prediction", "label").rdd.map(x =>
        (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double]))
    )
    println("MSE: " + rm.meanSquaredError)
    println("MAE: " + rm.meanAbsoluteError)
    println("RMSE Squared: " + rm.rootMeanSquaredError)
    println("R Squared: " + rm.r2)
    println("Explained Variance: " + rm.explainedVariance + "\n")*/

    val f1 = Figure()
    val p = f1.subplot(0)
   // val x: DenseVector[Double] = linspace(0.0, 1.0)//曲线可画区间
   // val x1=new DenseVector[Int](1 to 10 toArray)
    val x1: DenseVector[Double] = linspace(1, 10,10)
    val y1=new DenseVector(deepAUC)
    val y2=new DenseVector(deepAUC)
    p += plot(x1, y1)


    p.xlabel = "deep per tree"
    p.ylabel = "AUC"

   /* val f2 = Figure()
    val p2 = f2.subplot(0)
    // val x: DenseVector[Double] = linspace(0.0, 1.0)//曲线可画区间
    val  x2: DenseVector[Double] = linspace(10, 30,21)
    //val x2: DenseVector[Double] = linspace(10, 30)
    val y2=new DenseVector(treeAUC)
    p2 += plot(x2, y2)
    // p += plot(x, x :^ 3.0, '.')
    p2.xlabel = " number of the tree"
    p2.ylabel = "AUC"*/

   /* val f3 = Figure()
    val p3 = f3.subplot(0)
    // val x: DenseVector[Double] = linspace(0.0, 1.0)//曲线可画区间
    //val  x2: DenseVector[Double] = linspace(10, 30,21)
    //val x2: DenseVector[Double] = linspace(10, 30)
    //val y2=new DenseVector(treeAUC)
    p3 += image(td)
    // p += plot(x, x :^ 3.0, '.')
    p3.xlabel = " picture of AUC"
    p3.ylabel = "AUC"*/




  }
  def getAUC(deep:Int, treenum:Int, trainingData:DataFrame, testData:DataFrame): Double  ={
    val classifier = new RandomForestClassifier().setImpurity("gini").setMaxDepth(deep).setNumTrees(treenum).setFeatureSubsetStrategy("auto").setSeed(5043)
    val model = classifier.fit(trainingData)

    val evaluator = new BinaryClassificationEvaluator().setLabelCol("label")
    val predictions = model.transform(testData)
    // model.toDebugString
    //predictions.show
    //println(model.toDebugString)
    val accuracy = evaluator.evaluate(predictions) //compute result AUC
    val AUC=accuracy
    println("AUC is" + AUC)
    println("AUC is" + AUC)
    return AUC.toDouble

  }

  /*def getDecisionAUC(deep:Int,trainingData:DataFrame, testData:DataFrame):Double ={

  }*/

}

