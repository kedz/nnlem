
local BRNNSpanSelect, Parent = torch.class('nn.BRNNSpanSelect', 'nn.Module')

function BRNNSpanSelect:__init()
    self.forwardOutput = torch.Tensor()
    self.backwardOutput = torch.Tensor()
    self.gradForward = torch.Tensor()
    self.gradBackward = torch.Tensor()
    self.gradSegments = torch.Tensor()
end

function BRNNSpanSelect:updateOutput(input)
    local brnnOutputs = input[1]
    local segments = input[2]
    
    local dimSize = brnnOutputs[1]:size(3)
    local batchSize = segments:size(1)

    local forwardOutput = self.forwardOutput:resize(batchSize, dimSize)
    local backwardOutput = self.backwardOutput:resize(batchSize, dimSize)

    for batch=1,batchSize do
        local forwardIndex = segments[batch][2] - 1
        local backwardIndex = segments[batch][1]

        if forwardIndex ~= -1 then
            forwardOutput[batch]:copy(brnnOutputs[1][forwardIndex][batch])
            backwardOutput[batch]:copy(brnnOutputs[2][backwardIndex][batch])
        else
            forwardOutput[batch]:zero()
            backwardOutput[batch]:zero()
        end
    end

    self.output = {forwardOutput, backwardOutput}
    return self.output
end


function BRNNSpanSelect:updateGradInput(input, gradOutput)
    local brnnOutputs = input[1]
    local segments = input[2]
    local gradSegments = self.gradSegments:resizeAs(segments):zero()
    
    local dimSize = brnnOutputs[1]:size(3)
    local batchSize = segments:size(1)

    local forwardOutput, backwardOutput = table.unpack(self.output)
    local gradForward = self.gradForward:resizeAs(brnnOutputs[1]):zero()
    local gradBackward = self.gradBackward:resizeAs(brnnOutputs[2]):zero()

    for batch=1,batchSize do
        local forwardIndex = segments[batch][2] - 1
        local backwardIndex = segments[batch][1]

        if forwardIndex ~= -1 then
            gradForward[forwardIndex][batch]:add(gradOutput[1][batch])
            gradBackward[backwardIndex][batch]:add(gradOutput[2][batch])
        end
    end

    self.gradInput = {{gradForward, gradBackward}, gradSegments}
    return self.gradInput
end
