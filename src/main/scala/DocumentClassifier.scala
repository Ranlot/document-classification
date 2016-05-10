import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer, NGram, StopWordsRemover, VectorAssembler}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions.{udf, array}
import org.apache.log4j.{Level, Logger}
import language.postfixOps
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator

object DocumentClassifier {

  val makeVectorOfNumericalFeatures = udf((xs: Seq[Double]) => {
    val numbOfNumericalFeatures = xs.length
    Vectors.sparse(numbOfNumericalFeatures, (0 to (numbOfNumericalFeatures - 1)).toArray, xs.toArray)
  })

  def preProcess(line: String): String = {
    val extractedContent = line split "\t" drop 1 head
    val alphaNumericContent = extractedContent replaceAll("[^a-zA-Z0-9\\s]", "")
    alphaNumericContent toLowerCase
  }

  def getTweetID(line: String) = line split "\t" head

  def countSymbol(symbol: Char)(line: String) = line count (_ == symbol)

  def tweetLength(line: String) = line split " " length

  val countHashSymbol = countSymbol('#')(_)
  val countShtrudelSymbol = countSymbol('@')(_)
  val countExclamationSymbol = countSymbol('!')(_)

  def httpFinder(content: String) = content split " " count (_.startsWith("http"))

  val parseAndFindHTTP = httpFinder _ compose preProcess

  def wordFinder(word: String)(content: String) = content split " " count (_.contains(word))

  val winFinder = wordFinder("win")(_)
  val giveFinder = wordFinder("give")(_)
  val giveAwayFinder = wordFinder("giveaway")(_)
  val freeFinder = wordFinder("free")(_)

  case class Tweet(label: Double,
                   ID: String,
                   content: String,
                   numbOfWords: Double,
                   hashSignCounter: Double,
                   ShtrudelSignCounter: Double,
                   ExclaCounter: Double,
                   httpCounter: Double,
                   winCounter: Double,
                   giveCounter: Double,
                   giveAwayCounter: Double,
                   freeCounter: Double)

  def makeTweet(label: Double, line: String) = {
    Tweet(label,
      getTweetID(line),
      preProcess(line),
      tweetLength(line),
      countHashSymbol(line),
      countShtrudelSymbol(line),
      countExclamationSymbol(line),
      httpFinder(line),
      winFinder(line),
      giveFinder(line),
      giveAwayFinder(line),
      freeFinder(line)
    )
  }

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    val conf = new SparkConf().setMaster("local[*]").setAppName("documentClassification")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._

    // 1) read the data

    val goodTweets = sc.textFile("src/test/resources/good_non_marketing_tweets.shuffle.txt")
    val badTweets = sc.textFile("src/test/resources/bad_marketing_tweets.shuffle.txt")

    // 2) format as a tweet & calculate some numerical features

    val pos = goodTweets map {
      makeTweet(1.0, _)
    }
    val neg = badTweets map {
      makeTweet(0.0, _)
    }

    // 3) put data into dataframe

    var trainData = pos union neg toDF()

    // 4) join all numerical features into a single vector column

    val nameOfNumericalColumns = List("numbOfWords",
      "hashSignCounter",
      "ShtrudelSignCounter",
      "ExclaCounter",
      "httpCounter",
      "winCounter",
      "giveCounter",
      "giveAwayCounter",
      "freeCounter")

    val numericalColumns = nameOfNumericalColumns map {
      trainData(_)
    }

    trainData = trainData withColumn("numericalFeatures", makeVectorOfNumericalFeatures(array(numericalColumns: _*)))

    // 5) prepare a processing pipeline consisiting of:
    // tokenize text into list of words
    // calculate 2grams (keeping stop words)
    // remove stop words from list of words
    // tf-idf on hashed single words & hashed 2grams
    // assemble numerical features with both tf-idf all together into features column
    // apply logistic regression & cross-validation to get classification model

    val tokenizer = new Tokenizer() setInputCol "content" setOutputCol "words"
    val ngram = new NGram() setInputCol "words" setOutputCol "2grams" setN 2
    val remover = new StopWordsRemover() setInputCol "words" setOutputCol "filtered"

    val hash1 = new HashingTF() setInputCol "filtered" setOutputCol "hash1"
    val idf1 = new IDF() setInputCol "hash1" setOutputCol "idf1"

    val hash2 = new HashingTF() setInputCol "2grams" setOutputCol "hash2"
    val idf2 = new IDF() setInputCol "hash2" setOutputCol "idf2"

    val allAssembler = new VectorAssembler() setInputCols Array("numericalFeatures", "idf1", "idf2") setOutputCol "features"

    val lr = new LogisticRegression()

    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(0.9, 0.97))
      .addGrid(lr.maxIter, Array(100))
      .addGrid(hash1.numFeatures, Array(65536, 262144))
      .addGrid(hash2.numFeatures, Array(65536, 262144))
      .build()

    val pipeML = new Pipeline() setStages Array(tokenizer, ngram, remover, hash1, idf1, hash2, idf2, allAssembler, lr)
    val cv = new CrossValidator() setEstimator pipeML setEvaluator new BinaryClassificationEvaluator setEstimatorParamMaps paramGrid setNumFolds 3

    // 6) fit pipeline

    val model = cv fit trainData

    // 7) print cross-validation scores

    model.getEstimatorParamMaps.zip(model.avgMetrics).foreach(println)

    // 8) read new dataset and apply the model to make predictions

    val testTweets = sc.textFile("src/test/resources/tweets.shuffle.small.txt")
    //val testTweets = sc.textFile("/home/laurentb/NLP/tweets.shuffle.small.2.txt")
    //val testTweets = sc.textFile("/home/laurentb/NLP/tweets.shuffle.txt")

    var testTweetsDF = testTweets map {
      makeTweet(99.0, _) //the label here is irrelevant
    } toDF()

    val numericalTestColumns = nameOfNumericalColumns map {
      testTweetsDF(_)
    }

    testTweetsDF = testTweetsDF withColumn("numericalFeatures", makeVectorOfNumericalFeatures(array(numericalTestColumns: _*)))

    val allPreds = model transform testTweetsDF
    val myPreds = allPreds select("ID", "probability", "prediction")

    myPreds show()

    /*val res = myPreds.map(x => Array(x.getAs[String]("ID"), x.getAs[Vector]("probability").toArray(0).toString, x.getAs[Double]("prediction").toString).mkString(","))
    res.coalesce(1).saveAsTextFile("src/test/resources/result")*/

  }

}
