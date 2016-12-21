BASEPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Running setup script from $BASEPATH ..."
sleep 3

git clone https://github.com/torch/distro.git $BASEPATH/torch --recursive
cd $BASEPATH/torch
bash install.sh -s

. $BASEPATH/torch/install/bin/torch-activate

luarocks install torch
luarocks install nn
luarocks install dpnn
luarocks install torchx

if [[ $(which nvidia-smi) ]]; then
    echo "Installing gpu libs..."
    luarocks install cutorch
    luarocks install cunn
    luarocks install cunnx
else
    echo "Skipping gpu libs..."
fi

echo ". $BASEPATH/torch/install/bin/torch-activate" > $BASEPATH/env.sh
echo "export LUA_PATH=\"$LUA_PATH;$BASEPATH/lua/?.lua\"" >> $BASEPATH/env.sh
echo "export OMP_NUM_THREADS=1" >> $BASEPATH/env.sh

echo -e "\nSetting up data and experiments in $BASEPATH ..."

DATAPATH=$BASEPATH/data

 ### download data ###

mkdir -p $DATAPATH
cd $DATAPATH

if [ ! -f lemmatization-en.txt ]; then
    if [ ! -f lemmatization-en.zip ]; then
        wget http://www.lexiconista.com/Datasets/lemmatization-en.zip
    fi
    unzip lemmatization-en.zip
fi

cd $BASEPATH

python python/make-dataset.py --path $DATAPATH/lemmatization-en.txt \
    --seed 293492323491 --dest $DATAPATH/
