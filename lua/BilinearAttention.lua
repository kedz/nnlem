local BilinearAttention, Parent = torch.class('nn.BilinearAttention',
    'nn.Module')


function BilinearAttention:__init(temp)
    self.temp = temp
    self.isMasked = false
    self:buildOp(self.isMasked)

end

function BilinearAttention:setTemp(temp)
    self.temp = temp
    if self.tempLayer == nil then
        self:buildOp(self.isMasked)
        self.op:type(self:type())
    else
        self.tempLayer.constant_scalar = self.temp
    end

end

function BilinearAttention:maskZero()
    self.isMasked = true
    self:buildOp(self.isMasked)
    self.op:type(self:type())
    return self
end

function BilinearAttention:buildOp(mask)

    local softMax = mask and nn.SoftMaxMaskZero() or nn.SoftMax()

    self.op = nn.Sequential():add(
        nn.ConcatTable():add(
            nn.SelectTable(2)):add(
            nn.SelectTable(1))):add(
        nn.ParallelTable():add(
            nn.Unsqueeze(2)):add(
            nn.Transpose({1,3}, {1,2}))):add(
        nn.MM()):add(
        nn.Squeeze(2))
    if self.temp ~= nil and self.temp > 0 then
        self.tempLayer = nn.MulConstant(1/self.temp)
        self.op:add(self.tempLayer)
    end

    self.op:add(softMax)

end

function BilinearAttention:updateOutput(input)
    self.output = self.op:forward(input)
    return self.output
end

function BilinearAttention:updateGradInput(input, gradOutput)
    self.gradInput = self.op:backward(input, gradOutput)
    return self.gradInput
end
