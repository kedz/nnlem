pushd `dirname $0` > /dev/null
SCRIPTPATH=`pwd`
popd > /dev/null
cd $SCRIPTPATH/..
BASEPATH=`pwd`
cd $SCRIPTPATH

DATAPATH=$BASEPATH/data
TRAINPATH=$DATAPATH/lemmas.en.train.txt
DEVPATH=$DATAPATH/lemmas.en.dev.txt
TESTPATH=$DATAPATH/lemmas.en.test.txt

LUA_PATH_OLD=$LUA_PATH
LUA_PATH="$LUA_PATH;$BASEPATH/lua/?.lua"

th $BASEPATH/lua/eval-model.lua \
    --model $BASEPATH/alstm-models \
    --vocab $DATAPATH/vocab.txt \
    --train-data $TRAINPATH \
    --dev-data $DEVPATH \
    --batch-size 15 \
    --start-epoch 1 \
    --stop-epoch 50 \
    --seed 1986 \
    -
