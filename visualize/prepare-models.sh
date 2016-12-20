SHOW_TOP=5
BEAM_SIZE=3
BATCH_SIZE=5

mkdir -p metadata

th make-slstm-json.lua --model ../slstm-models \
    --batch-size $BATCH_SIZE \
    --show-top $SHOW_TOP \
    --beam-size $BEAM_SIZE \
    --vocab ../data/vocab.txt \
    --data ../data/lemmas.en.train.txt \
    --json metadata/slstm.v1.train.json

th make-slstm-json.lua --model ../slstm-models \
    --batch-size $BATCH_SIZE \
    --show-top $SHOW_TOP \
    --beam-size $BEAM_SIZE \
    --vocab ../data/vocab.txt \
    --data ../data/lemmas.en.dev.txt \
    --json metadata/slstm.v1.dev.json

th make-slstm-json.lua --model ../alstm-models \
    --batch-size $BATCH_SIZE \
    --show-top $SHOW_TOP \
    --beam-size $BEAM_SIZE \
    --vocab ../data/vocab.txt \
    --data ../data/lemmas.en.train.txt \
    --json metadata/alstm.v1.train.json

th make-slstm-json.lua --model ../alstm-models \
    --batch-size $BATCH_SIZE \
    --show-top $SHOW_TOP \
    --beam-size $BEAM_SIZE \
    --vocab ../data/vocab.txt \
    --data ../data/lemmas.en.dev.txt \
    --json metadata/alstm.v1.dev.json
