-- Options
local opt = lapp [[
Evaluate encoder/decoder model.
Options:
  --results     (string)        Path to write results.
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
  --progress   (default true)   Show progress bar.
]]

require 'lemma-data'
local eval = require('eval')
require 'stacked-lstm'
require 'attention-lstm'
require 'attention-bi-lstm'

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
    _, _, encInTrn, decInTrn, decOutTrn = data.read(
        opt.train_data, vocab, ids, opt.progress)
    nnzTrain = decInTrn:nonzero():size(1)
end

local encInDev, decInDev, decOutDev
local nnzDev
if string.len(opt.dev_data) > 0 then
    print("Reading development data from " .. opt.dev_data .. " ...")
    _, _, encInDev, decInDev, decOutDev = data.read(
        opt.dev_data, vocab, ids, opt.progress)
    nnzDev = decInDev:nonzero():size(1)
end

local encInTst, decInTst, decOutTst
local nnzTst
if string.len(opt.test_data) > 0 then
    print("Reading test data from " .. opt.test_data .. " ...")
    _, _, encInTst, decInTst, decOutTst = data.read(
        opt.test_data, vocab, ids, opt.progress)
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

local results = assert(io.open(opt.results, "w"))
header = "epoch"
if encInTrn then 
    header = header .. "\ttrain perpl\ttrain acc"
end
if encInDev then 
    header = header .. "\tdev perpl\tdev acc"
end
if encInTst then 
    header = header .. "\ttest perpl\ttest acc"
end
results:write(header .. "\n")

print("Evaluating epochs " .. opt.start_epoch .. " ... " .. opt.stop_epoch)

for epoch=opt.start_epoch,opt.stop_epoch do
    print("Epoch " .. epoch)

    local modelFile = opt.model .. "/model-" .. epoch .. ".bin"
    local model = torch.load(modelFile)
    if useGPU then model:cuda() end

    local trainLoss = 0
    local trainCorrect = 0
    if encInTrn then
        print("Running on training split.")
        for batch in data:batchIter(encInTrn, decInTrn, decOutTrn,
                                    opt.batch_size) do
            if opt.progress then
                xlua.progress(batch.t, batch.maxSteps)
            end
            local bEncInTrn = batch["encIn"]
            local bDecInTrn = batch["decIn"]
            local bDecOutTrn = batch["decOut"]
            local bLossTrn = model:loss(bEncInTrn, bDecInTrn, bDecOutTrn) 
            local bPredOutputTrn = model:greedyDecode(bEncInTrn, false, true)
            local bAccTrn = eval.coarseAccuracy(bPredOutputTrn, bDecOutTrn)
            local bNumCorrectTrn = bEncInTrn:size(1) * bAccTrn

            trainLoss = trainLoss + bLossTrn
            trainCorrect = trainCorrect + bNumCorrectTrn
        end
    end

    local devLoss = 0
    local devCorrect = 0
    if encInDev then
        print("Running on development split.")
        for batch in data:batchIter(encInDev, decInDev, decOutDev,
                                    opt.batch_size) do
            if opt.progress then
                xlua.progress(batch.t, batch.maxSteps)
            end
            local bEncInDev = batch["encIn"]
            local bDecInDev = batch["decIn"]
            local bDecOutDev = batch["decOut"]
            local bLossDev = model:loss(bEncInDev, bDecInDev, bDecOutDev) 
            local bPredOutputDev = model:greedyDecode(bEncInDev, false, true)
            local bAccDev = eval.coarseAccuracy(bPredOutputDev, bDecOutDev)
            local bNumCorrectDev = bEncInDev:size(1) * bAccDev

            devLoss = devLoss + bLossDev
            devCorrect = devCorrect + bNumCorrectDev
        end
    end

    local testLoss = 0
    local testCorrect = 0
    if encInTst then
        print("Running on test split.")
        for batch in data:batchIter(encInTst, decInTst, decOutTst,
                                    opt.batch_size) do
            if opt.progress then
                xlua.progress(batch.t, batch.maxSteps)
            end
            local bEncInTst = batch["encIn"]
            local bDecInTst = batch["decIn"]
            local bDecOutTst = batch["decOut"]
            local bLossTst = model:loss(bEncInTst, bDecInTst, bDecOutTst) 
            local bPredOutputTst = model:greedyDecode(bEncInTst, false, true)
            local bAccTst = eval.coarseAccuracy(bPredOutputTst, bDecOutTst)
            local bNumCorrectTst = bEncInTst:size(1) * bAccTst

            testLoss = testLoss + bLossTst
            testCorrect = testCorrect + bNumCorrectTst
        end
    end


    local resultString = '' .. epoch

    if encInTrn then
        trainPerpl = torch.exp(trainLoss / nnzTrain)
        trainAcc = trainCorrect / decInTrn:size(1)
        print(epoch, "   Training perplexity = " .. trainPerpl)
        print(epoch, "     Training accuracy = " .. trainAcc)
        resultString = resultString .. "\t" .. trainPerpl .. "\t" .. trainAcc
    end
    if encInDev then
        devPerpl = torch.exp(devLoss / nnzDev)
        devAcc = devCorrect / decInDev:size(1)
        print(epoch, "Development perplexity = " .. devPerpl)
        print(epoch, "  Development accuracy = " .. devAcc)
        resultString = resultString .. "\t" .. devPerpl .. "\t" .. devAcc
    end

    if encInTst then
        testPerpl = torch.exp(testLoss / nnzTst)
        testAcc = testCorrect / decInTst:size(1)
        print(epoch, "       Test perplexity = " .. testPerpl)
        print(epoch, "         Test accuracy = " .. testAcc)
        resultString = resultString .. "\t" .. testPerpl .. "\t" .. testAcc
    end

    results:write(resultString .. "\n")

end

results:close()
