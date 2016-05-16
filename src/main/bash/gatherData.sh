#!/usr/bin/env bash

dataSetName="smsspamcollection.zip"
dataDir="dataSet/"
expectedNumbOfFields=2

nameOfFile="SMSSpamCollection"

wget https://archive.ics.uci.edu/ml/machine-learning-databases/00228/$dataSetName
unzip $dataSetName -d $dataDir
rm $dataSetName

sed -i 's/\xc2\x92//g' $dataDir/$nameOfFile

#-----------------------------------------------

numberOfWrongLines=`cat $dataDir/$nameOfFile | awk -F '\t' '{if(NF!="'$expectedNumbOfFields'") print NR}' | wc -l`
possibleValues=`cat $dataDir/$nameOfFile | awk -F '\t' '{print $1}' | sort | uniq | tr '\n' ','`

if [ $numberOfWrongLines -ne 0 ]; then echo "custom error: Not the expected format."; exit 1; fi
if [ $possibleValues != "ham,spam," ]; then echo "custom error: Not the expected values"; exit 1; fi

function countValues {
  local countIs=`cat dataSet/SMSSpamCollection | awk '{if($1 == "'$1'") print $0}' | wc -l`
  echo $countIs
}

ham=$(countValues "ham")
spam=$(countValues "spam")

totMessages=`echo $ham + $spam | bc -l`
fromFileTotMessages=`wc -l < $dataDir/$nameOfFile`

if [ $totMessages -ne $fromFileTotMessages ]; then echo "custom error: Wrong number of messages"; exit 1; fi

spamRatio=`echo $spam / $totMessages | bc -l`

echo -e "ham = "$ham"\nspam = "$spam"\nspam ratio = "$spamRatio
#-----------------------------------------------
