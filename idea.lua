require 'nn'
require 'rnn'
require 'optim'
require 'BRNNSelect'
require 'MixtureTableV2'
require 'SoftMaxMaskZero'
require 'BilinearAttention'
require 'attention-bi-lstm'
local data = require('seg-lemma-data')

local voc, ids, encIn, decIn, decOut, segments = data.read("prefix-toy.dat")

print(data.tostring(encIn, segments, ids))

local dimSize = 128
local enc = nn.Sequential():add(
    nn.Transpose({2,1})):add(
    nn.LookupTableMaskZero(#ids, dimSize)):add(
    nn.SeqBRNN(dimSize, dimSize, false, nn.Identity()))
    
local selector = nn.Recursor(nn.Sequential():add(nn.BRNNSelect()):add(nn.CAddTable())):float()

enc = enc:float()

local dec = nn.Recursor(
    nn.Sequential():add(
        nn.LookupTableMaskZero(#ids, dimSize)):add(
        nn.FastLSTM(dimSize, dimSize):maskZero(1)))
dec = dec:float() 


local attend = nn.Recursor(nn.Sequential():add(
    nn.ConcatTable():add(
        nn.BilinearAttention():maskZero()
    ):add(
        nn.Sequential():add(nn.SelectTable(1)):add(nn.Transpose({2,1}))
    )
):add(
    nn.MixtureTableV2(2)
))
attend = attend:float()

local predict = nn.Recursor(
    nn.Sequential():add(
        nn.JoinTable(2)
    ):add(
        nn.MaskZero(nn.Linear(dimSize * 2, #ids), 1)
    ):add(nn.MaskZero(nn.LogSoftMax(),1 ))   
):float()


local nll = nn.ClassNLLCriterion():float()
nll.sizeAverage = false
local criterion = nn.SequencerCriterion(nn.MaskZeroCriterion(nll, 1)):float()
local tableOutput =
        nn.Sequential():add(
            nn.Transpose({2,1})
        ):add(
            nn.SplitTable(1)
        ):float()


local G = {}


function forward(encIn, decIn, segments)

    local batchSize = encIn:size(1)
    local H = enc:forward(encIn)
   
    G = {}
    for step=1,segments:size(2) - 1 do
        local segment = segments:narrow(2, step, 2)
        local Gstep = selector:forward({H, segment})
        table.insert(G, Gstep:view(1, batchSize, dimSize))

    end
   
    G = nn.JoinTable(1):float():forward(G)

    local output = {}
    for step=1,decIn:size(2) do 
        local y_in = decIn:select(2, step)
        local h_dec = dec:forward(y_in)
        local ghat = attend:forward({G, h_dec})
        table.insert(output, predict:forward({ghat, h_dec}))
    end

    return output
end

function backward(encIn, decIn, segments, gradOutput)

    local gradG = torch.FloatTensor():resizeAs(G):zero()

    for step=decIn:size(2),1,-1 do
        y_in = decIn:select(2, step)
        h_dec = dec:getStepModule(step).output

        local ghat = attend:getStepModule(step).output
        local gradPredict = predict:backward({ghat, h_dec}, gradOutput[step])
        local gradAttend = attend:backward({G, h_dec}, gradPredict[1])
        gradG:add(gradAttend[1])
        dec:backward(y_in, gradPredict[2] + gradAttend[2])
    end

    local H = enc.output
    local gradHfwd = torch.FloatTensor():resizeAs(H[1]):zero()
    local gradHbwd = torch.FloatTensor():resizeAs(H[2]):zero()

    for step=G:size(1),1,-1 do
        local gradG_t = gradG[step]
        local segment = segments:narrow(2, step, 2)
        local gradSelector = selector:backward({H, segment}, gradG_t)
        gradHfwd:add(gradSelector[1][1])
        gradHbwd:add(gradSelector[1][2])
    end

    enc:backward(encIn, {gradHfwd, gradHbwd})

end


local function appendTable(baseTable, otherTable)
    for i=1,#otherTable do table.insert(baseTable, otherTable[i]) end
end

local params = {}
local gradParams = {}

local eParams, eGradParams = enc:parameters()
appendTable(params, eParams)
appendTable(gradParams, eGradParams)
local dParams, dGradParams = dec:parameters()
appendTable(params, dParams)
appendTable(gradParams, dGradParams)

local pParams, pGradParams = predict:parameters()
appendTable(params, pParams)
appendTable(gradParams, pGradParams)

local parameters = nn.Module.flatten(params)
local gradParameters = nn.Module.flatten(gradParams)



local function feval(params)

    local reference = tableOutput:forward(decOut)
    gradParameters:zero()
    dec:forget()
    predict:forget()
    selector:forget()
    attend:forget()
    dec:forget()


    local lsm = forward(encIn, decIn, segments)
    local nll = criterion:forward(lsm, reference)
    local gradOutput = criterion:backward(lsm, reference)
    backward(encIn, decIn, segments, gradOutput)

    return nll, gradParameters
end

optimState = {learningRate=.01}
local model = nn.ABLSTMModel(#ids, dimSize, 1, .01)

for i=1,1000 do
    local _, loss = optim.adam(feval, parameters, optimState)
    local mloss = model:trainStep(encIn, decIn, decOut)
    print(i, loss[1], mloss)

end
    os.exit()

    local H = enc:forward(encIn)

print(H)
local G = selector:forward({H, segments:narrow(2,1,2)})
print(G)
