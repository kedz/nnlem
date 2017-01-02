SCRIPTPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPTPATH/..
BASEPATH=`pwd`

DATAPATH=$BASEPATH/data
TRAINPATH=$DATAPATH/lemmas.en.train.txt
DEVPATH=$DATAPATH/lemmas.en.dev.txt
TESTPATH=$DATAPATH/lemmas.en.test.txt

MODELS=`ls $BASEPATH/models`

mkdir -p $BASEPATH/results

parallel -j8 --progress --results /tmp/ev/ --joblog /tmp/ev-log \
    th $BASEPATH/lua/eval-model.lua \
    --results $BASEPATH/results/{1}.tsv \
    --model $BASEPATH/models/{1} \
    --vocab $DATAPATH/vocab.txt \
    --train-data $TRAINPATH \
    --dev-data $DEVPATH \
    --batch-size 50 \
    --start-epoch 1 \
    --stop-epoch 50 \
    --seed 1986 \
    --progress false \
    --gpu 0 ::: $MODELS
