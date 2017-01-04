local TruncPoissonPMF, Parent = torch.class('nn.TruncPoissonPMF', 'nn.Module')

function TruncPoissonPMF:__init(rightTrunc)
    self.rightTrunc = rightTrunc
    self.fact = torch.Tensor(rightTrunc)
    self.gradInput = torch.Tensor()

    self.fact[1] = 1
    for i=2,rightTrunc do
        self.fact[i] = self.fact[i-1] * i 
    end

    self.powerLayer = {}
    for i=1,rightTrunc do
        self.powerLayer[i] = nn.Sequential():add(
            nn.Power(i)):add(nn.MulConstant(1/self.fact[i]))
    end


    self.expLayer = nn.Sequential():add(nn.MulConstant(-1)):add(nn.Exp())

    self.join = nn.JoinTable(2)
    self.updf = nn.Sequential():add(nn.MM())
    self.normalize = nn.Sequential():add(nn.Sum(3)):add(
        nn.Unsqueeze(2)):add(
        nn.Power(-1))




end

function TruncPoissonPMF:updateOutput(input)

    local batchSize = input:size(1)
    local lambda = input

    self.lambdaPow = {}

    for i=1,self.rightTrunc do
        self.lambdaPow[i] = self.powerLayer[i]:forward(lambda):view(
            batchSize, 1)
    end
    
    local probi =  self.join:forward(self.lambdaPow)
    local expLam = self.expLayer:forward(lambda)

    if self.lastBatchSize ~= batchSize then

        self.adaptor = nn.ParallelTable():add(
            nn.View(batchSize, 1, 1)):add(
            nn.View(batchSize, 1, self.rightTrunc))

        self.pdf = nn.Sequential():add(
            self.adaptor
        ):add(
            self.updf
        ):add(
            nn.ConcatTable():add(self.normalize):add(nn.Identity())
        ):add(
            nn.MM()
        ):add(
            nn.Squeeze(2)
        )
        
        self.pdf:type(self:type())

        self.lastBatchSize = batchSize
    end

    self.output = self.pdf:forward({expLam, probi})
    return self.output

end

function TruncPoissonPMF:updateGradInput(input, gradOutput)

    local gradInput = self.gradInput:resizeAs(input):zero()
    local probi =  self.join.output
    local expLam = self.expLayer.output
    local gradPdf = self.pdf:backward({expLam, probi}, gradOutput)
    gradInput:add(self.expLayer:backward(input, gradPdf[1]))

    for i=1,self.rightTrunc do
        gradInput:add(
            self.powerLayer[i]:backward(input, gradPdf[2]:narrow(2, i, 1)))
    end

    return gradInput
end
