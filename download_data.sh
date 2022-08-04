# Download data


cd ./data/data_DisasterTweets/
kaggle competitions download -c nlp-getting-started
unzip nlp-getting-started.zip
rm nlp-getting-started.zip
rm sample_submission.csv
rm test.csv

cd ../data_IMDB/
curl https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz > aclImdb.tar.gz
tar -xvf aclImdb.tar.gz
rm -rf aclImdb.tar.gz

cd ../data_CIC_IDS2017/
kaggle datasets download -d cicdataset/cicids2017
unzip cicids2017.zip 
rm -fr cicids2017.zip 
cd MachineLearningCSV/MachineLearningCVE
mv * ../../
cd ../../
rm -rf MachineLearningCSV/

