require 'nn'
require 'rnn'
require 'optim'

local SLSTMModel = torch.class('nn.SLSTMModel')

function SLSTMModel:__init(vocabSize, dimSize, layerSize, learningRate)

    self.vocabSize = vocabSize
    self.dimSize = dimSize
    self.layerSize = layerSize
    self.optimState = {learningRate=learningRate}
    self:buildNet()
    self:__allocateMemory()
    self:reset()

end

function SLSTMModel:loss(encoderInput, decoderInput, decoderOutput)
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

function SLSTMModel:trainStep(encoderInput, decoderInput, decoderOutput)
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

function SLSTMModel:greedyDecode(encoderInput, returnScores, copy)
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
    self.encoder:forward(encoderInput)

    -- Pass lstm state from encoder to decoder.
    self:forwardConnect()

    -- Run decoder
    local totalSteps = 0
    for step=1,limit do
        totalSteps = step

        local logits_t = self.decoder:forward(decoderInput_t)
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

function SLSTMModel:forward(encoderInput, decoderInput)
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

    self.encoder:forward(encoderInput)

    -- Pass lstm state from encoder to decoder.
    self:forwardConnect()

    self.logits = {} 
    self.logProbability = {}
    for step=1,decoderSize do
        local decoderInput_t = decoderInput:select(2, step)
        local logits_t = self.decoder:forward(decoderInput_t)
        self.logits[step] = logits_t
        self.logProbability[step] = self.logSoftMax:forward(logits_t)
    end
    return self.logProbability
end

function SLSTMModel:backward(encoderInput, decoderInput, gradOutput)
--[[
--  Perform backward pass through the model. 
--
--  encoderInput -- batchSize x sequenceSize tensor of encoder inputs.
--  decoderInput -- batchSize x decoderSize tensor of decoder inputs.
--  gradOutput   -- decoderSize table of gradients for the log
--                  probability (i.e. log softmax) layer.
--]]

    local decoderSize = decoderInput:size(2)

    self.gradEncoderOutput = self.gradEncoderOutput or torch.Tensor()
    local gradEncoderOutput = self.gradEncoderOutput:typeAs(
        self.encoder.output):resizeAs(self.encoder.output)
    gradEncoderOutput:zero()

    self.gradLogProbability = {}
    for step=decoderSize,1,-1 do
        local gradDecoderOutput_t = self.logSoftMax:backward(
            self.logits[step], gradOutput[step])
        local decoderInput_t =  decoderInput:select(2, step)
        self.decoder:backward(decoderInput_t, gradDecoderOutput_t)
    end

    self:backwardConnect()
    self.encoder:backward(encoderInput, gradEncoderOutput)

end

function SLSTMModel:lossAndCoarseAcc(encoderInput, decoderInput, decoderOutput,
                                     normalize)

    print("WARNING: this method may not work.")
    if normalize == nil then
        normalize = true
    end

    local goldOutput = self.tableOutput:forward(decoderOutput)
    self:encoderForward(encoderInput)
    local logits = self:decoderForward(decoderInput)
    local nll = self:lossForward(logits, goldOutput)

    local seq = nn.Sequential():add(
        nn.MapTable():add(
            nn.Unsqueeze(1))
    ):add(
        nn.JoinTable(1)
    )
    seq = seq:type(logits[1]:type())

    logits = seq:forward(logits)
    logits = logits:permute(2,1,3)

    local _, argmax = torch.max(logits, 3)
    argmax = argmax:viewAs(decoderInput):typeAs(decoderInput)
    local mask = torch.eq(decoderInput, 0)
    argmax:maskedFill(mask, 0)

    local batchSize = decoderInput:size(1)
    local correct = 0
    for b=1,batchSize do
        if torch.all(torch.eq(argmax[b], decoderOutput[b])) then
            correct = correct + 1
        end
    end

    local acc = correct
    if normalize then
        acc = acc / batchSize
    end

    return nll, acc
end

function SLSTMModel:beamDecode(encoderInput, beamSize, 
                               outputScore, useOutputBuffer)

    print("WARNING: this method may not work.")
    if outputScore == nil then
        outputScore = false
    end

    if useOutputBuffer == nil then
        useOutputBuffer = false
    end

    local maxStepsOffset = 5
    local batchSize = encoderInput:size(1)
    local limit = encoderInput:size(2) + maxStepsOffset
    local isCuda = string.find(encoderInput:type(), "Cuda")

    self:forget()
    self.encoder:forward(encoderInput)

    local batchOutputTable = {}
    local batchOutputScores = {}
    for batch=1,batchSize do --batchSize do
 
        local beam = torch.LongTensor(beamSize, limit)
        local beamBuffer = torch.LongTensor(beamSize, limit)
        local cellBuffer = torch.FloatTensor(beamSize, self.dimSize)
        local outputBuffer = torch.FloatTensor(beamSize, self.dimSize)
        beamScores = torch.FloatTensor(beamSize)
        nextScores = torch.FloatTensor(beamSize * beamSize)

       

        self.decoder:forget()
        self:beamForwardConnect(batch, beamSize)
        beamScores:fill(0)

        beam:zero()
        beamBuffer:zero()

        local decoderInput_t = torch.LongTensor(beamSize):fill(2)


        local isFinished = torch.ByteTensor(beamSize):fill(0)

        local totalSteps
        for step=1,limit do
            totalSteps = step
            nextScores:zero()
            
            decoderInput_t:maskedFill(isFinished, 0)

            local logits_t = self.decoder:forward(decoderInput_t)
            local scores_t = self.logSoftMax:forward(logits_t)

            nextBeamScores, nextBeamOutputs = torch.topk(scores_t, beamSize, 2, true, true)
            for k=1,beamSize do
                if isFinished[k] == 1 then
                --if isFinished[k] == 1 then
                    nextScores_k = nextScores:narrow(
                        1, beamSize * (k - 1) + 1, beamSize)
                    nextScores_k:fill(-math.huge)
                else
                    nextScores_k = nextScores:narrow(
                        1, beamSize * (k - 1) + 1, beamSize)
                    nextScores_k:fill(beamScores[k])
                    nextScores_k:add(nextBeamScores[k])
                end
            end
            local flatBeamScores, flatBeamIndices = torch.sort(nextScores, 1, true) -- could be sort, when 
            

            -- this is bad -- breaks for stacked lstm
            local cell_t = self.decoder.module:get(2).cells[step]
            local output_t = self.decoder.module:get(2).outputs[step]

            found = 1
            used = {}
            flatBeamIndex = 1
            --for flatBeamIndex=1,flatBeamScores:size(1) do
            while flatBeamIndex <= flatBeamScores:size(1) and found <= beamSize do

            --for flatBeamIndex=1,beamSize do
                --while found <= beamSize and isFinished[found] do found = found + 1 end
                --if found > beamSize then break end

                if isFinished[found] == 1 then
                    beamBuffer[found]:narrow(1,1,step-1):copy(beam[found]:narrow(1,1,step-1))
                    found = found + 1 
                else

                    local i = math.ceil(flatBeamIndices[flatBeamIndex] / beamSize)
                    local j = (flatBeamIndices[flatBeamIndex] - 1) % beamSize + 1
                    local candidateOutput = nextBeamOutputs[i][j]
                    if used[candidateOutput] == nil then

                        local candidateScore = flatBeamScores[flatBeamIndex]
                      
                        beamScores[found] = candidateScore
                        if step > 1 then
                            beamBuffer[found]:narrow(1,1,step-1):copy(beam[i]:narrow(1,1,step-1))
                        end
                        beamBuffer[found][step] = nextBeamOutputs[i][j]
                        cellBuffer[found]:copy(cell_t[i])
                        outputBuffer[found]:copy(output_t[i])
                        if candidateOutput ~= 3 then
                        used[candidateOutput] = true
                        end
                        found = found + 1
                    end
                    --if found > beamSize then break end
                    flatBeamIndex = flatBeamIndex + 1
                end
            end


            cell_t:copy(cellBuffer)
            output_t:copy(outputBuffer)

            decoderInput_t:copy(beamBuffer:select(2, step))

            isFinished:maskedFill(torch.eq(decoderInput_t, 3), 1)


            local tmp = beam
            beam = beamBuffer
            beamBuffer = tmp

            if torch.all(isFinished) then 
                break 
            end

        end

        beam = beam:narrow(2,1,totalSteps)

        batchOutputTable[batch] = beam:clone()
        batchOutputScores[batch] = beamScores:clone()
        for b=1,beamSize do
            batchOutputScores[batch][b] = batchOutputScores[batch][b] / beam[b]:nonzero():size(1)
        end
        
        local scores, indices = torch.sort(batchOutputScores[batch], true)

        batchOutputTable[batch] =  batchOutputTable[batch]:index(1, indices)
        batchOutputScores[batch] = scores

    end
    
    return batchOutputTable, batchOutputScores
end

    ------------------------- LAYER ACCESSORS ------------------------------

function SLSTMModel:getEncoderLSTM(layer)

    if layer == nil then
        layers = {}
        for l=1,self.layerSize do
            table.insert(layers, self.encoder:get(2 + l))
        end
        return layers
    else
        assert(layer > 0 and layer <= self.layerSize, "Arg #1 out of range.")
        return self.encoder:get(2 + layer)
    end

end

function SLSTMModel:getDecoderLSTM(layer)

    if layer == nil then
        layers = {}
        for l=1,self.layerSize do
            table.insert(layers, self.decoder.module:get(1 + l))
        end
        return layers
    else
        assert(layer > 0 and layer <= self.layerSize, "Arg #1 out of range.")
        return self.decoder.module:get(1 + layer)
    end

end
    ------------------------- RECURRENT UTILS ------------------------------

function SLSTMModel:forget()
--[[
--  Reset recurrent operations to the first timestep. Call this before 
--  starting to predict a sequence or do a forward pass.
--]]
    self.encoder:forget()
    self.decoder:forget()
    self.logSoftMax:forget()
end

function SLSTMModel:forwardConnect()
--[[
--  Pass state and output of last encoder lstm step to the first decoder lstm
--  step for each lstm layer. Call this after the forward pass of the encoder.
--]]
    for l=1,self.layerSize do
        local enc = self:getEncoderLSTM(l)
        local dec = self:getDecoderLSTM(l)
        dec.userPrevOutput = nn.rnn.recursiveCopy(
            dec.userPrevOutput, enc.output[-1])
        dec.userPrevCell = nn.rnn.recursiveCopy(
            dec.userPrevCell, enc.cell[-1])

    end
end

function SLSTMModel:backwardConnect()
--[[
--  Pass gradient of state and output of first decoder lstm step to the 
--  last encoder lstm step for each lstm layer. Call this after the backward 
--  pass of the decoder.
--]]
    for l=self.layerSize,1,-1 do
        local enc = self:getEncoderLSTM(l)
        local dec = self:getDecoderLSTM(l)
        enc.userNextGradCell = nn.rnn.recursiveCopy(
            enc.userNextGradCell, dec.userGradPrevCell)
        enc.gradPrevOutput = nn.rnn.recursiveCopy(
            enc.gradPrevOutput, dec.userGradPrevOutput)

    end
end

function SLSTMModel:beamForwardConnect(batch, beamSize)
    for l=1,self.layerSize do
        local enc = self:getEncoderLSTM(l)
        local dec = self:getDecoderLSTM(l)
        local encOutput = enc.output[-1][batch]:view(
            1, self.dimSize):expand(beamSize, self.dimSize)
        local encCell = enc.cell[-1][batch]:view(
            1, self.dimSize):expand(beamSize, self.dimSize)
        dec.userPrevOutput = encOutput
        dec.userPrevCell = encCell
    end
end

    -------------------------   INIT NETWORK  ------------------------------
    
function SLSTMModel:__allocateMemory()
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

function SLSTMModel:buildNet()

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

function SLSTMModel:buildEncoder()
    local encoder = nn.Sequential():add(
            nn.Transpose({2,1})
        ):add(
            nn.LookupTableMaskZero(self.vocabSize, self.dimSize)
        )

    for l=1,self.layerSize do
        local lstm = nn.SeqLSTM(self.dimSize, self.dimSize)
        lstm:maskZero(1)
        encoder:add(lstm)
    end
    return encoder

end

function SLSTMModel:buildDecoder()
    local decoder_step = nn.Sequential()
    decoder_step:add(nn.LookupTableMaskZero(self.vocabSize, self.dimSize))
    for l=1,self.layerSize do
        local lstm = nn.FastLSTM(self.dimSize, self.dimSize)
        lstm:maskZero(1)
        decoder_step:add(lstm)
    end
    decoder_step:add(nn.MaskZero(nn.Linear(self.dimSize, self.vocabSize), 1))

    return nn.Recursor(decoder_step)
end

    -------------------------   MISC METHODS  ------------------------------

function SLSTMModel:float()
    self.encoder:float()
    self.decoder:float()
    self.logSoftMax:float()
    self.criterion:float()
    self.tableOutput:float()
    self.__allocateMemory()
end

function SLSTMModel:cuda()
    self.encoder:cuda()
    self.decoder:cuda()
    self.logSoftMax:cuda()
    self.criterion:cuda()
    self.tableOutput:cuda()
    self.__allocateMemory()
end

function SLSTMModel:reset()
    self:forget()
    self.encoder:reset()
    self.decoder:reset()
    self:zeroGradParameters()
end

function SLSTMModel:zeroGradParameters()
    self.encoder:zeroGradParameters()
    self.decoder:zeroGradParameters()
end
