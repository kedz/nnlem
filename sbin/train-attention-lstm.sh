SCRIPTPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPTPATH/..
BASEPATH=`pwd`

DATAPATH=$BASEPATH/data
TRAINPATH=$DATAPATH/lemmas.en.train.txt
DEVPATH=$DATAPATH/lemmas.en.dev.txt
TESTPATH=$DATAPATH/lemmas.en.test.txt

DIMS="256 128 64 32"
LAYERS="3 2 1"
LRS="1E-3 1E-2 1E-4"
BSS="15 25 50"

parallel -j4 --progress --results /tmp/alstm/ --joblog /tmp/alstm-log \
    th $BASEPATH/lua/train-attention-lstm.lua \
    --vocab $DATAPATH/vocab.txt \
    --data $TRAINPATH \
    --dims {1} \
    --layers {2} \
    --lr {3} \
    --batch-size {4} \
    --save $BASEPATH/models/alstm.dim{1}.layers{2}.lr{3}.bs{4} \
    --progress false ::: $DIMS ::: $LAYERS ::: $LRS ::: $BSS
    

