pushd `dirname $0` > /dev/null
SCRIPTPATH=`pwd`
popd > /dev/null

echo "Setting up data and experiments in $SCRIPTPATH ..."

DATAPATH=$SCRIPTPATH/data

### download data ###

mkdir -p $DATAPATH

cd $DATAPATH

if [ ! -f lemmatization-en.txt ]; then
    if [ ! -f lemmatization-en.zip ]; then
        wget http://www.lexiconista.com/Datasets/lemmatization-en.zip
    fi
    unzip lemmatization-en.zip
fi

cd $SCRIPTPATH

python python/make-dataset.py --path $DATAPATH/lemmatization-en.txt \
    --seed 293492323491 --dest $DATAPATH/
