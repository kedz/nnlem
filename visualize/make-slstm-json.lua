-- Options
local opt = lapp [[
Extract metadata from a model for visualization.
Options:
  --model       (string)        Path to model/vocab directory.
  --vocab       (string)        Path to vocab file.
  --data        (string)        Training data path.
  --json        (string)        Path of json file
  --batch-size  (default 15)    Batch size. 
  --show-top    (default 15)    Show top k greedy predictions.
  --beam-size   (default 5)     Size of beam for beam search.
  --stop-epoch  (default 50)    Last epoch to evaluate (inclusive).
  --seed        (default 1986)  Random seed.
  --gpu         (default 0)     Which gpu to use. Default is cpu.
]]

require 'lemma-data'
require 'stacked-lstm'
require 'attention-lstm'
local json = require('cjson')

--[[ 
--  Add batch inputs and gold outputs to metadata. 
--]]
function addExamples(metadata, encIn, decIn, decOut, ids, dataPath)
    metadata["examples"] = {}
    metadata["path"] = dataPath
    local batchSize = encIn:size(1)
    for b=1,batchSize do
        local encoderInput = {}
        local goldDecoderInput = {}
        local goldDecoderOutput = {}
        for t=1,encIn:size(2) do 
            if encIn[b][t] > 0 then
                table.insert(encoderInput, ids[encIn[b][t]])
            end
        end
        for t=1,decIn:size(2) do 
            if decIn[b][t] > 0 then
                table.insert(goldDecoderInput, ids[decIn[b][t]])
                table.insert(goldDecoderOutput, ids[decOut[b][t]])
            end
        end
        local examplesMetadata = {}
        examplesMetadata["encoderInput"] = encoderInput
        examplesMetadata["goldDecoderInput"] = goldDecoderInput
        examplesMetadata["goldDecoderOutput"] = goldDecoderOutput
        examplesMetadata["models"] = {}
        table.insert(metadata["examples"], examplesMetadata)
    end
end

--[[ 
--  Add model information to the metadata table.
--]]
function addModelMetadata(metadata, model)
    modelMetadata = {}
    modelMetadata["dimSize"] = model.dimSize
    modelMetadata["layerSize"] = model.layerSize
    modelMetadata["type"] = torch.type(model)
    if torch.type(model) == 'nn.SLSTMModel' then
        modelMetadata["attention"] = false
    else
        modelMetadata["attention"] = true
    end
    metadata["model"] = modelMetadata
end

--[[
--  Get model attention metadata.
--]]
local function getAttentionMetadata(model, example, maxSteps, encIn, decOut)
    local attention = {}
    local inputSize = encIn:size(2)
    for step=1,maxSteps do
        if decOut[example][step] > 0 then
            local attention_step = {}
            for i=1,inputSize do
                if encIn[example][i] > 0 then
                    table.insert(
                            attention_step,
                            model:getAttentionStep(step).output[example][i])
                end
            end
            table.insert(attention, attention_step)
        end
    end
    return attention
end

--[[
--  Get metadata about greedy predicted output steps.
--]]
local function getGreedyDecoderOutput(model, ex, encIn, decOut, ids, showTop)
    local inputSize = encIn:size(1)
    local maxSteps = decOut:size(2)
    local predictions = {}
    for step=1,maxSteps do 
        if decOut[ex][step] > 0 then
            local predStep = {}
            local lp = model.logProbability[step][ex]
            local scores, tokens = torch.topk(lp, showTop, 1, true, true)
            for candidate=1,showTop do
                local token = ids[tokens[candidate]]
                local probability = torch.exp(scores[candidate])
                table.insert(predStep, {token=token, probability=probability})
            end
            table.insert(predictions, predStep)
        end
    end
    return predictions
end

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

print("Reading data from " .. opt.data .. " ...")
local _, _, encIn, decIn, decOut = data.read(opt.data, vocab, ids)

if useGPU then
    encIn = encIn:cuda()
    decIn = decIn:cuda()
    decOut = decOut:cuda()
end

local bEncIn, bDecIn, bDecOut
for batch in data:batchIter(encIn, decIn, decOut, opt.batch_size) do
    bEncIn = batch["encIn"]:clone()
    bDecIn = batch["decIn"]:clone()
    bDecOut = batch["decOut"]:clone()
    break
end

local metadata = {}
addExamples(metadata, bEncIn, bDecIn, bDecOut, ids, opt.data)

for epoch=1,opt.stop_epoch do
    xlua.progress(epoch, opt.stop_epoch)

    local modelFile = opt.model .. "/model-" .. epoch .. ".bin"
    local model = torch.load(modelFile)

    if metadata["model"] == nil then 
        addModelMetadata(metadata, model)
    end

    local greedyOutput = model:greedyDecode(bEncIn, false, false, true)

    for example=1,opt.batch_size do
        local maxSteps = greedyOutput:size(2)
        local predictionMetadata = {}

        predictionMetadata["greedyDecoderOutput"] = getGreedyDecoderOutput(
            model, example, bEncIn, greedyOutput, ids, opt.show_top)

        if metadata["model"]["attention"] then
            predictionMetadata["attention"] = getAttentionMetadata(
                model, example, maxSteps, bEncIn, greedyOutput)
        end

        table.insert(
            metadata["examples"][example]["models"],
            predictionMetadata
        )
    end

    local beamOutput, beamScores = model:beamDecode(
        bEncIn, opt.beam_size, true)
    
    for example=1,opt.batch_size do
        beamDecoderOutput = {}
        for beam=1,opt.beam_size do
            local outputString = data.tostring(beamOutput[example][beam], ids)
            local score = torch.exp(beamScores[example][beam])
            table.insert(
                beamDecoderOutput, 
                {output=outputString, score=score})
        end
        metadata["examples"][example]["models"][epoch]["beamDecoderOutput"] = beamDecoderOutput
    end
    
end

local f = assert(io.open(opt.json, "w"))
print("Writing metadata to " .. opt.json .. " ...")
f:write(json.encode(metadata))
f:close()
