-- Options
local opt = lapp [[
Evaluate a stacked LSTM encoder/decoder.
Options:
  --model       (string)        Path to model/vocab directory.
  --vocab       (string)        Path to vocab file.
  --train-data  (default '')    Training data path.
  --dev-data    (default '')    Development data path.
  --test-data   (default '')    Test data path.
  --batch-size  (default 15)    Batch size. 
  --start-epoch (default 1)     Starting epoch to evaluate.
  --stop-epoch  (default 50)    Last epoch to evaluate (inclusive).
  --seed        (default 1986)  Random seed.
  --gpu         (default 0)     Which gpu to use. Default is cpu.
]]

require 'lemma-data'
require 'stacked-lstm'
require 'attention-lstm'

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

print("Reading vocab from " .. opt.vocab .. " ...")
local vocab, ids = data.readVocab(opt.vocab)

local encInTrn, decInTrn, decOutTrn
local nnzTrain
if string.len(opt.train_data) > 0 then
    print("Reading training data from " .. opt.train_data .. " ...")
    _, _, encInTrn, decInTrn, decOutTrn = data.read(opt.train_data, vocab, ids)
    nnzTrain = decInTrn:nonzero():size(1)
end

local encInDev, decInDev, decOutDev
local nnzDev
if string.len(opt.dev_data) > 0 then
    print("Reading development data from " .. opt.dev_data .. " ...")
    _, _, encInDev, decInDev, decOutDev = data.read(opt.dev_data, vocab, ids)
    nnzDev = decInDev:nonzero():size(1)
end

local encInTst, decInTst, decOutTst
local nnzTst
if string.len(opt.test_data) > 0 then
    print("Reading test data from " .. opt.test_data .. " ...")
    _, _, encInTst, decInTst, decOutTst = data.read(opt.test_data, vocab, ids)
    nnzTst = decInTst:nonzero():size(1)
end

if useGPU then
    if encInTrn then
        encInTrn = encInTrn:cuda()
        decInTrn = decInTrn:cuda()
        decOutTrn = decOutTrn:cuda()
    end
    if encInDev then
        encInDev = encInDev:cuda()
        decInDev = decInDev:cuda()
        decOutDev = decOutDev:cuda()
    end
    if encInTst then
        encInTst = encInTst:cuda()
        decInTst = decInTst:cuda()
        decOutTst = decOutTst:cuda()
    end
else
    if encInTrn then
        encInTrn = encInTrn:float()
        decInTrn = decInTrn:float()
        decOutTrn = decOutTrn:float()
    end
    if encInDev then
        encInDev = encInDev:float()
        decInDev = decInDev:float()
        decOutDev = decOutDev:float()
    end
    if encInTst then
        encInTst = encInTst:float()
        decInTst = decInTst:float()
        decOutTst = decOutTst:float()
    end
end

print("Evaluating epochs " .. opt.start_epoch .. " ... " .. opt.stop_epoch)

for epoch=opt.start_epoch,opt.stop_epoch do
    print("Epoch " .. epoch)

    local modelFile = opt.model .. "/model-" .. epoch .. ".bin"
    local model = torch.load(modelFile)

    local trainLoss = 0
    local trainCorrect = 0
    if encInTrn then
        for batch in data:batchIter(encInTrn, decInTrn, decOutTrn,
                                    opt.batch_size) do
            xlua.progress(batch.t, batch.maxSteps)
            local bEncInTrn = batch["encIn"]
            local bDecInTrn = batch["decIn"]
            local bDecOutTrn = batch["decOut"]
            local loss, correct = model:lossAndCoarseAcc(
                bEncInTrn, bDecInTrn, bDecOutTrn, false)
            trainLoss = trainLoss + loss
            trainCorrect = trainCorrect + correct
        end
    end

    local devLoss = 0
    local devCorrect = 0
    if encInDev then
        for batch in data:batchIter(encInDev, decInDev, decOutDev,
                                    opt.batch_size) do
            xlua.progress(batch.t, batch.maxSteps)
            local bEncInDev = batch["encIn"]
            local bDecInDev = batch["decIn"]
            local bDecOutDev = batch["decOut"]
            local loss, correct = model:lossAndCoarseAcc(
                bEncInDev, bDecInDev, bDecOutDev, false)
            devLoss = devLoss + loss
            devCorrect = devCorrect + correct
        end
    end

    if encInTrn then
        trainPerpl = torch.exp(trainLoss / nnzTrain)
        trainAcc = trainCorrect / decInTrn:size(1)
        print(epoch, "  Training perplexity = " .. trainPerpl)
        print(epoch, "    Training accuracy = " .. trainAcc)
    end
    if encInDev then
        devPerpl = torch.exp(devLoss / nnzDev)
        devAcc = devCorrect / decInDev:size(1)
        print(epoch, "Development perplexity = " .. devPerpl)
        print(epoch, "  Development accuracy = " .. devAcc)
    end


end
