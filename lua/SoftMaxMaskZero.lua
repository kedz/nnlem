local SoftMaxMaskZero, Parent = torch.class('nn.SoftMaxMaskZero', 'nn.Module')

function SoftMaxMaskZero:__init()

    self.inputMasked = torch.Tensor()
    self.gradOutputMasked = torch.Tensor()
    self.op = nn.SoftMax()
    self.mask = nil

end

function SoftMaxMaskZero:updateOutput(input)

    self.inputMasked:resizeAs(input):copy(input)
    self.mask = torch.eq(input, 0)
    self.inputMasked:maskedFill(self.mask, -math.huge)
    self.output = self.op:forward(self.inputMasked)
    self.output:maskedFill(self.mask, 0)

    return self.output

end

function SoftMaxMaskZero:updateGradInput(input, gradOutput)
    self.gradOutputMasked:resizeAs(gradOutput):copy(gradOutput)
    self.gradOutputMasked:maskedFill(self.mask, 0)
    self.gradInput = self.op:backward(self.inputMasked, self.gradOutputMasked)
    return self.gradInput
end
