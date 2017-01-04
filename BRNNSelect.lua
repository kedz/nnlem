
local BRNNSelect, Parent = torch.class('nn.BRNNSelect', 'nn.Module')

function BRNNSelect:__init()
    self.forwardOutput = torch.Tensor()
    self.backwardOutput = torch.Tensor()
    self.gradForward = torch.Tensor()
    self.gradBackward = torch.Tensor()
    self.gradSegments = torch.Tensor()
end

function BRNNSelect:updateOutput(input)
    local brnnOutputs = input[1]
    local indices = input[2]
    
    local seqSize = brnnOutputs[1]:size(1)
    local dimSize = brnnOutputs[1]:size(3)
    local batchSize = indices:size(1)

    local forwardOutput = self.forwardOutput:resize(batchSize, dimSize)
    local backwardOutput = self.backwardOutput:resize(batchSize, dimSize)

    for batch=1,batchSize do
        local index = indices[batch]
        if index > 0 and index <= seqSize then
            forwardOutput[batch]:copy(brnnOutputs[1][index][batch])
            backwardOutput[batch]:copy(brnnOutputs[2][index][batch])
        else
            forwardOutput[batch]:zero()
            backwardOutput[batch]:zero()
        end
    end

    self.output = {forwardOutput, backwardOutput}
    return self.output
end


