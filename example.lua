require 'rnn'
require 'attention-lstm'

-- start encoder token = 1
-- start decoder token = 2
-- stop decoder token = 3

local encIn =  torch.FloatTensor{{  1,  4,  5,  6},  {  1,  7,  8,  9}}
local decOut = torch.FloatTensor{{ 10, 11, 12,  3},  { 13, 14, 15,  3}}
local decIn =  torch.FloatTensor{{  2, 10, 11, 12},  {  2, 13, 14, 15}}

local dimSize = 32
local layers = 1
local learningRate = .001
local vocabSize = 15

print("Attention LSTM with dimension size 32, and 1 layer.")

local model = nn.ALSTMModel(vocabSize, dimSize, layers, learningRate)

for i=1,1000 do
    local loss = model:trainStep(encIn, decIn, decOut)
    if i % 100 == 0 then
        print(i, loss)
    end
end

local loss = model:loss(encIn, decIn, decOut)
print(loss)


print("gold")
print("====")
print(decOut)
print("predicted")
print("=========")
decOutPred, logProb = model:greedyDecode(encIn, true) -- arg 2 makes function return predicted output AND log prob of predictions.
print(decOutPred)
print("Probability")
print("===========")
print(torch.exp(logProb))
