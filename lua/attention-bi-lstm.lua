require 'nn'
require 'rnn'
require 'optim'
require 'BilinearAttention'
require 'SoftMaxMaskZero'
require 'MixtureTableV2'

local ABLSTMModel = torch.class('nn.ABLSTMModel')

function ABLSTMModel:__init(vocabSize, dimSize, layerSize, learningRate)

    self.vocabSize = vocabSize
    self.dimSize = dimSize
    self.layerSize = layerSize
    self.learningRate = learningRate
    self.optimState = {learningRate=learningRate}
    self:buildNet()
    self:__allocateMemory()
    self:reset()

end

function ABLSTMModel:loss(encoderInput, decoderInput, decoderOutput)
--[[ 
--  Compute negative log likelihood of decoderOutput given encoderInput and 
--  decoderInput. 
--  
--  encoderInput -- batchSize x sequenceSize tensor of encoder inputs.
--  decoderInput -- batchSize x sequenceSize tensor of decoder inputs.
--  decoderOutput -- batchSize x sequenceSize tensor of decoder outputs.
--]]

    -- Criterion expects table input.
    local reference = self.tableOutput:forward(decoderOutput)

    -- Reset recurrent opt to first step.
    self:forget()

    -- Forward pass to get loss (negative log likelihood)
    local logProbability = self:forward(encoderInput, decoderInput)
    local nll = self.criterion:forward(logProbability, reference)
    return nll
end

function ABLSTMModel:trainStep(encoderInput, decoderInput, decoderOutput)
--[[ 
--  Perform a gradient descent step given encoder/decoder inputs and outputs.
--  Currently using the adamax optimizer.
--  Returns negative log likelihood from forward pass.
--
--  encoderInput -- batchSize x sequenceSize tensor of encoder inputs.
--  decoderInput -- batchSize x sequenceSize tensor of decoder inputs.
--  decoderOutput -- batchSize x sequenceSize tensor of decoder outputs.
--]]

    local function feval(params)

        -- Criterion expects table input.
        local reference = self.tableOutput:forward(decoderOutput)

        -- Zero grad parameters and reset recurrent ops to first step.
        self:zeroGradParameters()
        self:forget()

        -- Forward
        local logProbability = self:forward(encoderInput, decoderInput)
        local nll = self.criterion:forward(logProbability, reference)

        -- Backward
        local gradOutput = self.criterion:backward(logProbability, reference)
        self:backward(encoderInput, decoderInput, gradOutput)

        return nll, self.gradParameters

    end

    local _, loss = optim.adamax(feval, self.parameters, self.optimState)
    return loss[1]

end

function ABLSTMModel:greedyDecode(encoderInput, returnScores, copy)
--[[ 
--  Predict a batch of output sequences given a batch of input sequences.
--  Prediction is done using the greedy best prediction at each time step.
--
--  encoderInput -- a batchSize x sequenceSize zero padded input tensor.
--                  e.g. torch.FloatTensor{{0,0,1,2,3},
--                                         {0,1,4,2,5}}
--  returnScores -- boolean flag, when true, function also returns a batchSize
--                  vector of log probabilities of each predicted sequences.
--                  Default equals false.
--  copy -- boolean flag, when true, new memory is allocated for the predicted
--          outputs and optional scores. When false, this memory buffer is 
--          reused for subsequent predictions. Set to false if doing some
--          batch processing and outputs are not needed after subsequent 
--          predictions.
--          Default equals true.
--
--  Returns batchSize x predicted length tensor of predicted outputs,
--          (optional) batchSize length vector of scores
--]]

    if returnScores == nil then returnScores = false end
    if copy == nil then copy = true end

    self.logProbability = {}
    local maxStepsOffset = 5

    local batchSize = encoderInput:size(1)
    local limit = encoderInput:size(2) + maxStepsOffset
    local isCuda = string.find(encoderInput:type(), "Cuda")

    ------------------------- ALLOCATE MEMORY ------------------------------

    -- outputs contains predicted outputs, i.e. the result of greedy decoding.
    -- Max generatable output is the size of input sequence + maxStepsOffset.
    local outputs
    if useOutputBuffer then
        self.outputBuffer = self.outputBuffer or torch.Tensor()
        if isCuda then
            self.outputBuffer = self.outputBuffer:type(
                "torch.CudaLongTensor")
        else
            self.outputBuffer = self.outputBuffer:long()
        end
        outputs = self.outputBuffer:resize(limit, batchSize):zero()

    else
        if isCuda then
            outputs = torch.CudaLongTensor():resize(limit, batchSize):zero()
        else
            outputs = torch.LongTensor():resize(limit, batchSize):zero()
        end
    end

    -- isFinished[i] = 1 when this batch item has produced the stop
    -- token. When torch.all(isFinished) is true, we can 
    -- stop generating tokens.
    self.isFinishedBuffer = self.isFinishedBuffer or torch.Tensor()
    if isCuda then
        self.isFinishedBuffer = self.isFinishedBuffer:type(
            "torch.CudaByteTensor")
    else
        self.isFinishedBuffer = self.isFinishedBuffer:byte()
    end
    local isFinished = self.isFinishedBuffer:resize(batchSize, 1):fill(0)

    -- Setup decoderInput_t memory and fill it with the start decoding token 2.
    self.decoderInputBuffer = self.decoderInputBuffer or torch.Tensor()
    self.decoderInputBuffer = self.decoderInputBuffer:typeAs(outputs)
    local decoderInput_t = self.decoderInputBuffer:resize(batchSize):fill(2)

    -- Setup location to keep logits or log prob of the predicted outputs.
    self.scoresBuffer = self.scoresBuffer or torch.Tensor()
    self.scoresBuffer = self.scoresBuffer:type(self.decoder:type())

    local scores = self.scoresBuffer:resize(
        limit, batchSize):zero()

    self:forget()

    -------------------------- RUN DECODER --------------------------------

    -- Encoder Forward 
    local encoderOutput = self.encoder:forward(encoderInput)

    -- Pass lstm state from encoder to decoder.
    self:forwardConnect()

    -- Run decoder
    local totalSteps = 0
    for step=1,limit do
        totalSteps = step

        local logits_t = self.decoder:forward({encoderOutput, decoderInput_t})
        local allScores_t = self.logSoftMax:forward(logits_t)
        self.logProbability[step] = allScores_t

        local scores_t, outputs_t = torch.max(
            scores[step], outputs[step],
            allScores_t, 2)
        outputs_t:maskedFill(isFinished, 0)

        local isStopToken = torch.eq(outputs_t, 3)
        isFinished:maskedFill(isStopToken, 1)

        decoderInput_t:copy(outputs_t:view(batchSize)):maskedFill(
            isFinished, 0)

        if torch.all(isFinished) then break end

    end

    ------------------------ RETURN OUTPUT AND SCORES ---------------------

    outputs = outputs:t():narrow(2,1,totalSteps)

    if returnScores then
        self.returnScoreBuffer = self.returnScoreBuffer or torch.Tensor()
        self.returnScoreBuffer = self.returnScoreBuffer:typeAs(scores)

        scores = scores:t():narrow(2,1,totalSteps)
        local score = torch.sum(self.returnScoreBuffer, scores, 2)
        score = score:view(batchSize) 

        if copy then
            return outputs:clone(), score:clone()
        else
            return outputs, score
        end
    else
        if copy then
            return outputs:clone()
        else
            return outputs
        end
    end
end

function ABLSTMModel:forward(encoderInput, decoderInput)
--[[
--  Perform a forward pass through the model using encoderInput and
--  decoderInput. Returns table log probability 
--  
--  encoderInput -- batchSize x sequenceSize tensor of encoder inputs.
--  decoderInput -- batchSize x sequenceSize tensor of decoder inputs.
--
--  Returns a table of log probabilities with decoderInput:size(2) entries.
--  Each entry is a batchSize by vocabSize matrix of log probabilities.
--]]

    local decoderSize = decoderInput:size(2)

    local encoderOutput = self.encoder:forward(encoderInput)

    -- Pass lstm state from encoder to decoder.
    self:forwardConnect()
    
    self.logits = {} 
    self.logProbability = {}
    for step=1,decoderSize do
        local decoderInput_t = {encoderOutput, decoderInput:select(2, step)}
        local logits_t = self.decoder:forward(decoderInput_t)
        self.logits[step] = logits_t
        self.logProbability[step] = self.logSoftMax:forward(logits_t)
    end
    return self.logProbability
end

function ABLSTMModel:backward(encoderInput, decoderInput, gradOutput)
--[[
--  Perform backward pass through the model. 
--
--  encoderInput -- batchSize x sequenceSize tensor of encoder inputs.
--  decoderInput -- batchSize x decoderSize tensor of decoder inputs.
--  gradOutput   -- decoderSize table of gradients for the log
--                  probability (i.e. log softmax) layer.
--]]

    local decoderSize = decoderInput:size(2)
    local encoderOutput = self.encoder.output

    self.gradEncoderOutput = self.gradEncoderOutput or torch.Tensor()
    self.gradEncoderOutput = self.gradEncoderOutput:typeAs(encoderOutput)
    local gradEncoderOutput = self.gradEncoderOutput:resizeAs(encoderOutput)
    gradEncoderOutput:zero()

    self.gradLogProbability = {}
    for step=decoderSize,1,-1 do
        local gradDecoderOutput_t = self.logSoftMax:backward(
            self.logits[step], gradOutput[step])
        local decoderInput_t = {encoderOutput, decoderInput:select(2, step)}
        local gradDecoder_t = self.decoder:backward(
            decoderInput_t, gradDecoderOutput_t)
        gradEncoderOutput:add(gradDecoder_t[1])
    end

    self:backwardConnect()
    self.encoder:backward(encoderInput, gradEncoderOutput)

end

    ------------------------- LAYER ACCESSORS ------------------------------

function ABLSTMModel:getForwardEncoderLSTM(layer)

    if layer == nil then
        layers = {}
        for l=1,self.layerSize do
            table.insert(
                layers, self.encoder:get(2 + l).forwardModule)
        end
        return layers
    else
        assert(layer > 0 and layer <= self.layerSize, "Arg #1 out of range.")
        return self.encoder:get(2 + layer).forwardModule
    end

end

function ABLSTMModel:getDecoderLSTM(layer)

    local mod = self.decoder.module:get(1):get(2)
    if layer == nil then
        layers = {}
        for l=1,self.layerSize do
            table.insert(layers, mod:get(1 + l))
        end
        return layers
    else
        assert(layer > 0 and layer <= self.layerSize, "Arg #1 out of range.")
        return mod:get(1 + layer)
    end

end

function ABLSTMModel:getAttentionStep(step)
    return self.decoder:getStepModule(step):get(2):get(1):get(1):get(1)
end

    ------------------------- RECURRENT UTILS ------------------------------
    
function ABLSTMModel:forget()
--[[
--  Reset recurrent operations to the first timestep. Call this before 
--  starting to predict a sequence or do a forward pass.
--]]
    self.encoder:forget()
    self.decoder:forget()
    self.logSoftMax:forget()
end

function ABLSTMModel:forwardConnect()
--[[
--  Pass state and output of last encoder lstm step to the first decoder lstm
--  step for each lstm layer. Call this after the forward pass of the encoder.
--]]

    local encoderSize = self.encoder.output:size(1)

    for l=1,self.layerSize do
        local enc = self:getForwardEncoderLSTM(l)
        local dec = self:getDecoderLSTM(l)
        dec.userPrevOutput = nn.rnn.recursiveCopy(
            dec.userPrevOutput, enc.output[-1])
        dec.userPrevCell = nn.rnn.recursiveCopy(
            dec.userPrevCell, enc.cell[-1])
    end
end

function ABLSTMModel:backwardConnect()
--[[
--  Pass gradient of state and output of first decoder lstm step to the 
--  last encoder lstm step for each lstm layer. Call this after the backward 
--  pass of the decoder.
--]]
    for l=1,self.layerSize do
        local enc = self:getForwardEncoderLSTM(l)
        local dec = self:getDecoderLSTM(l)
        enc.userNextGradCell = nn.rnn.recursiveCopy(
            enc.userNextGradCell, dec.userGradPrevCell)
        enc.gradPrevOutput = nn.rnn.recursiveCopy(
            enc.gradPrevOutput, dec.userGradPrevOutput)

    end
end

--function ALSTMModel:beamForwardConnect(batch, beamSize)
--    for l=1,self.layerSize do
--        local enc = self:getEncoderLSTM(l)
--        local dec = self:getDecoderLSTM(l)
--        local encOutput = enc.output[-1][batch]:view(
--            1, self.dimSize):expand(beamSize, self.dimSize)
--        local encCell = enc.cell[-1][batch]:view(
--            1, self.dimSize):expand(beamSize, self.dimSize)
--        dec.userPrevOutput = encOutput
--        dec.userPrevCell = encCell
--    end
--end


    -------------------------   INIT NETWORK  ------------------------------
    
function ABLSTMModel:__allocateMemory()
--[[
--  Set parameters and parameter gradients of encoder and decoder networks to 
--  be views of a single flat memory storage. This is necessary for training 
--  with the optim package. This needs to be called whenever the type of the 
--  network is changed.
--]]
--
    local function appendTable(baseTable, otherTable)
        for i=1,#otherTable do table.insert(baseTable, otherTable[i]) end
    end

    local params = {}
    local gradParams = {}

    local eParams, eGradParams = self.encoder:parameters()
    appendTable(params, eParams)
    appendTable(gradParams, eGradParams)
    local dParams, dGradParams = self.decoder:parameters()
    appendTable(params, dParams)
    appendTable(gradParams, dGradParams)

    self.parameters = nn.Module.flatten(params)
    self.gradParameters = nn.Module.flatten(gradParams)

end

function ABLSTMModel:buildNet()

    self.encoder = self:buildEncoder()
    self.decoder = self:buildDecoder()
    self.logSoftMax = nn.Recursor(nn.MaskZero(nn.LogSoftMax(), 1))

    local nll = nn.ClassNLLCriterion()
    nll.sizeAverage = false
    self.criterion = nn.SequencerCriterion(nn.MaskZeroCriterion(nll, 1))

    -- Convert gold outputs from batch x steps tensor to a table with steps
    -- entries of batch output tokens. This is necessary for the 
    -- SequencerCriterion since the output of the logSoftMax layer is also
    -- a table.
    self.tableOutput =
        nn.Sequential():add(
            nn.Transpose({2,1})
        ):add(
            nn.SplitTable(1)
        )

    self.encoder:float()
    self.decoder:float()
    self.logSoftMax:float()
    self.tableOutput:float()
    self.criterion:float()

end

function ABLSTMModel:buildEncoder()
    local encoder = nn.Sequential():add(
            nn.Transpose({2,1})
        ):add(
            nn.LookupTableMaskZero(self.vocabSize, self.dimSize)
        )

    for l=1,self.layerSize do
        local lstm = nn.SeqBRNN(self.dimSize, self.dimSize)
        lstm.forwardModule:maskZero(1)
        lstm.backwardModule:maskZero(1)
        encoder:add(lstm)
    end


    return encoder

end

function ABLSTMModel:buildDecoder()
    local decoder_step = nn.Sequential()
    decoder_step:add(
        nn.ParallelTable():add(nn.Identity()):add(self:buildDecoderInput())
    )
    decoder_step:add(
        nn.ConcatTable():add(self:buildAttention()):add(nn.SelectTable(2))
    )
    decoder_step:add(nn.JoinTable(2))
    decoder_step:add(nn.MaskZero(nn.Linear(self.dimSize*2, self.vocabSize), 1))
    return nn.Recursor(decoder_step)
end

function ABLSTMModel:buildDecoderInput()
    local decoder_input_step = nn.Sequential():add(
            nn.LookupTableMaskZero(self.vocabSize, self.dimSize)
        )
    for l=1,self.layerSize do
        local lstm = nn.FastLSTM(self.dimSize, self.dimSize)
        lstm:maskZero(1)
        decoder_input_step:add(lstm)
    end
        
    return decoder_input_step
end

function ABLSTMModel:buildAttention()
    local attention = nn.Sequential()
    attention:add(
        nn.ConcatTable():add(
            nn.BilinearAttention():maskZero()
        ):add(
            nn.Sequential():add(nn.SelectTable(1)):add(nn.Transpose({2,1}))
        )
    ):add(
        nn.MixtureTableV2(2)
    )
    return attention
end

    -------------------------   MISC METHODS  ------------------------------

function ABLSTMModel:float()
    self.encoder:float()
    self.decoder:float()
    self.logSoftMax:float()
    self.criterion:float()
    self.tableOutput:float()
    self:__allocateMemory()
end

function ABLSTMModel:cuda()
    self.encoder:cuda()
    self.decoder:cuda()
    self.logSoftMax:cuda()
    self.criterion:cuda()
    self.tableOutput:cuda()
    self:__allocateMemory()
end

function ABLSTMModel:cloneAsFloat() 
    local floatModel = nn.ABLSTMModel(
        self.vocabSize, self.dimSize, self.layerSize, self.learningRate)
    floatModel.encoder = self.encoder:clone():float()
    floatModel.decoder = self.decoder:clone():float()
    floatModel:__allocateMemory()
    return floatModel
end

function ABLSTMModel:reset()
    self:forget()
    self.encoder:reset()
    self.decoder:reset()
    self:zeroGradParameters()
end

function ABLSTMModel:zeroGradParameters()
    self.encoder:zeroGradParameters()
    self.decoder:zeroGradParameters()
end
