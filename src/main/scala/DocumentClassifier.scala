import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer, NGram, StopWordsRemover, VectorAssembler}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions.{udf, array}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator

object DocumentClassifier {

  val makeVectorOfNumericalFeatures = udf((xs: Seq[Double]) => {
    val numbOfNumericalFeatures = xs.length
    Vectors.sparse(numbOfNumericalFeatures, (0 to (numbOfNumericalFeatures - 1)).toArray, xs.toArray)
  })

  def preProcess(line: String): String = {
    val extractedContent = line.split("\t").drop(1)(0)
    val alphaNumericContent = extractedContent.replaceAll("[^a-zA-Z0-9\\s]", "")
    alphaNumericContent
  }

  def getTweetID(line: String) = line.split("\t")(0)

  def countSymbol(symbol: Char)(line: String) = line.count(_ == symbol)

  def tweetLength(line: String) = line.split(" ").length

  def countHashSymbol = countSymbol('#')(_)
  def countShtrudelSymbol = countSymbol('@')(_)
  def countExclamationSymbol = countSymbol('!')(_)

  case class Tweet(label: Double, ID: String, content: String, numbOfWords: Double, hashSignCounter: Double, ShtrudelSignCounter: Double, ExclaCounter: Double)

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    val conf = new SparkConf().setMaster("local[*]").setAppName("documentClassification")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._

    // 1) read the data

    /*val goodTweets = sc.textFile("/home/laurentb/NLP/good_non_marketing_tweets.shuffle.txt")
    val badTweets = sc.textFile("/home/laurentb/NLP/bad_marketing_tweets.shuffle.txt")*/

    val goodTweets = sc.textFile("src/test/resources/good_non_marketing_tweets.shuffle.txt")
    val badTweets = sc.textFile("src/test/resources/bad_marketing_tweets.shuffle.txt")

    // 2) format as a tweet & calculate some numerical features

    val pos = goodTweets.map(line => Tweet(1.0, getTweetID(line), preProcess(line), tweetLength(line), countHashSymbol(line), countShtrudelSymbol(line), countExclamationSymbol(line)))
    val neg = badTweets.map(line => Tweet(0.0, getTweetID(line), preProcess(line), tweetLength(line), countHashSymbol(line), countShtrudelSymbol(line), countExclamationSymbol(line)))

    // 3) put data into dataframe

    var trainData = pos.union(neg).toDF()

    // 4) join all numerical features into a single vector column

    val numericalColumns = List("numbOfWords", "hashSignCounter", "ShtrudelSignCounter", "ExclaCounter") map { trainData(_) }
    trainData = trainData withColumn("numericalFeatures", makeVectorOfNumericalFeatures(array(numericalColumns: _*)))

    // 5) prepare a processing pipeline consisiting of:
    // tokenize text into list of words
    // calculate 2grams (keeping stop words)
    // remove stop words from list of words
    // tf-idf on hashed single words & hashed 2grams
    // assemble numerical features with both tf-idf all together into features column
    // apply logistic regression & cross-validation to get classification model

    val tokenizer = new Tokenizer().setInputCol("content").setOutputCol("words")
    val ngram = new NGram().setInputCol("words").setOutputCol("2grams").setN(2)
    val remover = new StopWordsRemover().setInputCol("words").setOutputCol("filtered")

    val hash1 = new HashingTF().setInputCol("filtered").setOutputCol("hash1").setNumFeatures(4096)
    val idf1 = new IDF().setInputCol("hash1").setOutputCol("idf1")

    val hash2 = new HashingTF().setInputCol("2grams").setOutputCol("hash2").setNumFeatures(4096)
    val idf2 = new IDF().setInputCol("hash2").setOutputCol("idf2")

    val allAssembler = new VectorAssembler().setInputCols(Array("numericalFeatures", "idf1", "idf2")).setOutputCol("features")

    val lr = new LogisticRegression()

    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(0.3, 0.5, 0.7, 0.9))
      .addGrid(lr.maxIter, Array(100))
      .addGrid(hash1.numFeatures, Array(16384))
      .addGrid(hash2.numFeatures, Array(16384))
      .build()


    val pipeML = new Pipeline().setStages(Array(tokenizer, ngram, remover, hash1, idf1, hash2, idf2, allAssembler, lr))
    val cv = new CrossValidator().setEstimator(pipeML).setEvaluator(new BinaryClassificationEvaluator).setEstimatorParamMaps(paramGrid).setNumFolds(3)

    // 6) fit pipeline

    val model = cv.fit(trainData)

    // 7) print cross-validation scores

    model.getEstimatorParamMaps.zip(model.avgMetrics).foreach(println)

    // 8) read new dataset and apply the model to make predictions

    //val testTweets = sc.textFile("/home/laurentb/NLP/tweets.shuffle.small.txt")
    val testTweets = sc.textFile("src/test/resources/tweets.shuffle.small.txt")
    //val testTweets = sc.textFile("/home/laurentb/NLP/tweets.shuffle.small.2.txt")
    //val testTweets = sc.textFile("/home/laurentb/NLP/tweets.shuffle.txt")

    val formatTestTweets = testTweets.map(line => Tweet(1.0, getTweetID(line), preProcess(line), tweetLength(line), countHashSymbol(line), countShtrudelSymbol(line), countExclamationSymbol(line))).toDF()

    val numericalColumnsTest = List("numbOfWords", "hashSignCounter", "ShtrudelSignCounter", "ExclaCounter") map { formatTestTweets(_) }
    val formatTestTweetsReady = formatTestTweets withColumn("numericalFeatures", makeVectorOfNumericalFeatures(array(numericalColumnsTest: _*)))

    val allPreds = model.transform(formatTestTweetsReady)
    val myPreds = allPreds.select("ID", "probability", "prediction")

    val res = myPreds.map(x => Array(x.getAs[String]("ID"), x.getAs[Vector]("probability").toArray(0).toString, x.getAs[Double]("prediction").toString).mkString(","))
    res.coalesce(1).saveAsTextFile("src/test/resources/result")

  }

}
