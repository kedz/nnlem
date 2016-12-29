-- Options
local opt = lapp [[
Train a stacked LSTM encoder/decoder.
Options:
  --vocab      (string)        Vocab path.
  --data       (string)        Training data path.
  --save       (default '')    Directory to write models/vocab. 
                               Default is no write.
  --batch-size (default 15)    Batch size. 
  --dims       (default 32)    Embedding/lstm dimension.
  --layers     (default 1)     Number of stacked lstm layers.
  --lr         (default 1E-3)  Learning rate.
  --epochs     (default 50)    Max number of training epochs.
  --seed       (default 1986)  Random seed.
  --gpu        (default 0)     Which gpu to use. Default is cpu.
  --progress   (default true)  Show progress bar.
]]

require 'lemma-data'
require 'attention-lstm'
require 'lfs'

local useGPU = false
if opt.gpu > 0 then useGPU = true end

if useGPU then
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(opt.gpu)
    cutorch.manualSeed(opt.seed)
    torch.manualSeed(opt.seed)
    print("running on gpu-" .. opt.gpu)

else
    torch.manualSeed(opt.seed)
    print("running on cpu")
end

local saveModels = string.len(opt.save) > 0

if saveModels then
    local result, msg = lfs.mkdir(opt.save)
    if result == nil and msg ~= "File exists" then
        print(msg)
        os.exit()
    end
else
    print("WARNING: not saving models to file!")
end

-------------------------------------------------------------------------------
------------------------------   READING DATA    ------------------------------
-------------------------------------------------------------------------------

print("Reading vocab from " .. opt.vocab .. " ...")
local vocab, ids = data.readVocab(opt.vocab)

print("Reading data from " .. opt.data .. " ...")
local _, _, encIn, decIn, decOut = data.read(
    opt.data, vocab, ids, opt.progress)

-------------------------------------------------------------------------------
------------------------------    SETUP MODEL    ------------------------------
-------------------------------------------------------------------------------

local vocabSize = #ids
local nnz = decIn:nonzero():size(1)

local model = nn.ALSTMModel(vocabSize, opt.dims, opt.layers, opt.lr)
if useGPU then
    encIn = encIn:cuda()
    decIn = decIn:cuda()
    decOut = decOut:cuda()
    model:cuda()
end

-------------------------------------------------------------------------------
------------------------------     TRAINING      ------------------------------
-------------------------------------------------------------------------------

print("\nTraining model for " .. opt.epochs .. " epochs... \n")

for epoch=1,opt.epochs do
    print("Epoch " .. epoch .. " ... ")
    local trainLoss = 0
    for batch in data:batchIter(encIn, decIn, decOut, opt.batch_size) do
        if opt.progress then
            xlua.progress(batch.t, batch.maxSteps)
        end
        bEncIn = batch["encIn"]
        bDecIn = batch["decIn"]
        bDecOut = batch["decOut"]

        local loss = model:trainStep(bEncIn, bDecIn, bDecOut)
        trainLoss = trainLoss + loss

    end

    print(epoch, "avg perplexity = " .. torch.exp(trainLoss / nnz))
    if saveModels then
        local modelFile = opt.save .. "/model-" .. epoch .. ".bin"
        print("Writing model to " .. modelFile .. " ...")
        torch.save(modelFile, model)
    end
end
