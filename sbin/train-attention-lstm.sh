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

th $BASEPATH/lua/train-attention-lstm.lua --data $TRAINPATH \
    --dims 64 \
    --vocab $DATAPATH/vocab.txt --layers 1 \
    --save $BASEPATH/alstm-models
