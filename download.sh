#!/bin/bash

#this downloads the zip file that contains the data
curl http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip -O -J -L --output trainingandtestdata.zip
# this unzips the zip file - you will get a directory named "data" containing the data
unzip trainingandtestdata.zip
# this cleans up the zip file, as we will no longer use it
rm trainingandtestdata.zip

echo downloaded data