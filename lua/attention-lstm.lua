require 'nn'
require 'rnn'
require 'optim'
require 'BilinearAttention'
require 'SoftMaxMaskZero'

local ALSTMModel = torch.class('nn.ALSTMModel')

function ALSTMModel:__init(vocabSize, dimSize, layerSize, learningRate)

    self.vocabSize = vocabSize
    self.dimSize = dimSize
    self.layerSize = layerSize
    self.optimState = {learningRate=learningRate}
    self:buildNet()
    self:__allocateMemory()
    self:reset()

end

function ALSTMModel:cuda()
    self.encoder:cuda()
    self.decoder:cuda()
    self.logSoftMax:cuda()
    self.criterion:cuda()
    self.tableOutput:cuda()
    self.__allocateMemory()
end

function ALSTMModel:reset()
    self.encoder:reset()
    self.decoder:reset()
    self:zeroGradParameters()
end

function ALSTMModel:zeroGradParameters()
    self.encoder:zeroGradParameters()
    self.decoder:zeroGradParameters()
end

function ALSTMModel:__allocateMemory()
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

function ALSTMModel:buildNet()

    self.encoder = nn.Sequential():add(
        nn.Transpose({2,1})
    ):add(
        nn.LookupTableMaskZero(self.vocabSize, self.dimSize)
    )

    for l=1,self.layerSize do
        local lstm = nn.SeqLSTM(self.dimSize, self.dimSize)
        lstm:maskZero(1)
        self.encoder:add(lstm)
    end

    local decoder_input_step = nn.Sequential():add(
            nn.LookupTableMaskZero(self.vocabSize, self.dimSize)
        )
    for l=1,self.layerSize do
        local lstm = nn.FastLSTM(self.dimSize, self.dimSize)
        lstm:maskZero(1)
        decoder_input_step:add(lstm)
    end

    local decoder_step = 
        nn.Sequential():add(
            nn.ParallelTable():add(
                nn.Identity()
            ):add(
                decoder_input_step)
        )

    decoder_step:add(
        nn.ConcatTable():add(
            nn.Sequential():add(
                nn.ConcatTable():add(
                    nn.BilinearAttention():maskZero()
                ):add(
                    nn.Sequential():add(
                        nn.SelectTable(1)
                    ):add(
                        nn.Transpose({2,1})
                    )
                )
            ):add(
                nn.MixtureTable(2)
            )
        ):add(
            nn.SelectTable(2)
        )
    ):add(
        nn.JoinTable(2)
    )


    decoder_step:add(nn.MaskZero(nn.Linear(self.dimSize*2, self.vocabSize), 1))

    self.decoder = nn.Recursor(decoder_step)

    self.logSoftMax = nn.Recursor(nn.MaskZero(nn.LogSoftMax(), 1))

    local nll = nn.ClassNLLCriterion()
    nll.sizeAverage = false
    self.criterion = nn.SequencerCriterion(nn.MaskZeroCriterion(nll, 1))

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

function ALSTMModel:getAttentionStep(step)
    return self.decoder:getStepModule(step):get(2):get(1):get(1):get(1)
end


function ALSTMModel:getEncoderLSTM(layer)

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

function ALSTMModel:getDecoderLSTM(layer)

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

function ALSTMModel:forget()
    self.encoder:forget()
    self.decoder:forget()
    self.logSoftMax:forget()
end

function ALSTMModel:forwardConnect()
    for l=1,self.layerSize do
        local enc = self:getEncoderLSTM(l)
        local dec = self:getDecoderLSTM(l)
        dec.userPrevOutput = enc.output[-1]
        dec.userPrevCell = enc.cell[-1]
    end
end

function ALSTMModel:beamForwardConnect(batch, beamSize)
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

function ALSTMModel:backwardConnect()
    for l=1,self.layerSize do
        local enc = self:getEncoderLSTM(l)
        local dec = self:getDecoderLSTM(l)
        enc.gradPrevOutput = dec.userGradPrevOutput
        enc.userNextGradCell = dec.userGradPrevCell
    end
end

function ALSTMModel:encoderForward(encoderInput)
    self:forget()
    local output = self.encoder:forward(encoderInput)
    self:forwardConnect()
    return output
end

function ALSTMModel:encoderBackward(encoderInput, gradEncoderOutput)
    self.encoder:backward(encoderInput, gradEncoderOutput)

end

function ALSTMModel:decoderForward(encoderOutput, decoderInput)
    maxSteps = decoderInput:size(2)
    self.logitsOutput = {}
    local Y = decoderInput:t()
    for step=1,maxSteps do
        local Yt = Y[step]
        self.logitsOutput[step] = self.decoder:forward({encoderOutput, Yt})
    end
    return self.logitsOutput
end

function ALSTMModel:decoderBackward(encoderOutput, decoderInput, gradOutput)
    self.gradEncoderOutput = self.gradEncoderOutput or torch.Tensor()
    self.gradEncoderOutput = self.gradEncoderOutput:typeAs(encoderOutput)
    local gradEncoderOutput = self.gradEncoderOutput:resizeAs(encoderOutput)
    gradEncoderOutput:zero()

    local maxSteps
    local Y
    if type(decoderInput) == "table" then
        maxSteps = #decoderInput
        Y = decoderInput
    else
        maxSteps = decoderInput:size(2)
        Y = decoderInput:t()
    end
    for step=maxSteps,1,-1 do
        local gradStep = self.decoder:backward(
            {encoderOutput, Y[step]}, gradOutput[step])
        gradEncoderOutput:add(gradStep[1])
    end
    self:backwardConnect()
    return gradEncoderOutput
end

function ALSTMModel:lossForward(logits, output)
    local stepSize = #logits
--    local batchSize = logits[1]:size(1)
--    local vocabSize = logits[1]:size(2)
    
--    self.logProbability = self.logProbability or torch.Tensor()

--    local logProbability = self.logProbability:typeAs(logits[1]):resize(
--        stepSize, batchSize, vocabSize)

    self.logProbability = {}
    for step=1,stepSize do
        self.logProbability[step] = self.logSoftMax:forward(logits[step]) 
    end
    local nll = self.criterion:forward(self.logProbability, output)
    return nll
end

function ALSTMModel:lossBackward(logits, output)
    local stepSize = #logits
    --local logProbability = self.logSoftMax.output
    self.gradLogProbability = {}
    local gradOutput = self.criterion:backward(self.logProbability, output)
    
    for step=stepSize,1,-1 do
        self.gradLogProbability[step] = self.logSoftMax:backward(
            logits[step], gradOutput[step])
    end

    return self.gradLogProbability
end

function ALSTMModel:trainStep(encoderInput, decoderInput, decoderOutput)
    local goldOutput = self.tableOutput:forward(decoderOutput)
    self:zeroGradParameters()
    local function feval(params)
        local encoderOutput = self:encoderForward(encoderInput)
        local logits = self:decoderForward(encoderOutput, decoderInput)
        local nll = self:lossForward(logits, goldOutput)
        local gradOutput = self:lossBackward(logits, goldOutput)
        local gradEncoderOutput = self:decoderBackward(
            encoderOutput, decoderInput, gradOutput)
        self:encoderBackward(encoderInput, gradEncoderOutput)
        return nll, self.gradParameters
    end
    local _, loss = optim.adamax(feval, self.parameters, self.optimState)
    return loss[1]

end

function ALSTMModel:loss(encoderInput, decoderInput, decoderOutput)
    local goldOutput = self.tableOutput:forward(decoderOutput)
    local encoderOutput = self:encoderForward(encoderInput)
    local logits = self:decoderForward(encoderOutput, decoderInput)
    local nll = self:lossForward(logits, goldOutput)
    return nll
end

function ALSTMModel:lossAndCoarseAcc(encoderInput, decoderInput, decoderOutput,
                                     normalize)

    if normalize == nil then
        normalize = true
    end

    local goldOutput = self.tableOutput:forward(decoderOutput)
    local encoderOutput = self:encoderForward(encoderInput)
    local logits = self:decoderForward(encoderOutput, decoderInput)
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

function ALSTMModel:greedyDecode(encoderInput, outputScore, useOutputBuffer,
                                 computeLogSoftMax)
--[[ 
       encoderInput    -- a batchSize x sequenceSize zero padded input tensor.
       outputScore     -- optional boolean flag (defaults to false), 
                          when true, also outputs unnormalized log likelihood 
                          of each output sequence generated by the decoder.
       useOutputBuffer -- optional boolean flag (defaults to false),
                          when true, output tensor is reused by subsequent
                          calls to this function. Only set to true if you 
                          finish using the output before the next call to 
                          greedyDecoder, e.g. when batch processing.
--]]

    if outputScore == nil then
        outputScore = false
    end

    if useOutputBuffer == nil then
        useOutputBuffer = false
    end

    if computeLogSoftMax == nil then
        computeLogSoftMax = false
    end

    if computeLogSoftMax then
        self.logProbability = {}
    end

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
    self.isFinishedBuffer = self.isFinishedBuffer:byte()
    local isFinished = self.isFinishedBuffer:resize(batchSize, 1):fill(0)

    -- Setup decoderInput_t memory and fill it with the start decoding token 2.
    self.decoderInputBuffer = self.decoderInputBuffer or torch.Tensor()
    self.decoderInputBuffer = self.decoderInputBuffer:typeAs(outputs)
    local decoderInput_t = self.decoderInputBuffer:resize(batchSize):fill(2)

    -- Setup location to keep logits or log prob of the predicted outputs.
    self.scoresBuffer = self.scoresBuffer or torch.Tensor()
    if string.find(encoderInput:type(), "Cuda") then
        self.scoresBuffer = self.scoresBuffer:cuda()
    else
        self.scoresBuffer = self.scoresBuffer:float()
    end
    local scores = self.scoresBuffer:resize(
        limit, batchSize):zero()

    -------------------------- RUN DECODER --------------------------------

    -- Encode input
    local encoderOutput = self:encoderForward(encoderInput)

    -- Run decoder
    local totalSteps = 0
    for step=1,limit do
        totalSteps = step

        local logits_t = self.decoder:forward({encoderOutput, decoderInput_t})

        local allScores_t
        if computeLogSoftMax then
            allScores_t = self.logSoftMax:forward(logits_t)
            self.logProbability[step] = allScores_t
        else
            allScores_t = logits_t
        end

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

    if outputScore then
        scores = scores:t():narrow(2,1,totalSteps)
        local score = scores:sum(2):view(batchSize)
        return outputs, score
    else
        return outputs
    end
end

function ALSTMModel:beamDecode(encoderInput, beamSize, 
                               outputScore, useOutputBuffer)

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
    local encoderOutput = self.encoder:forward(encoderInput)

    local batchOutputTable = {}
    local batchOutputScores = {}
    for batch=1,batchSize do --batchSize do
 
        local encoderOutput_b = encoderOutput:narrow(2,batch, 1):expand(
            encoderOutput:size(1), beamSize, encoderOutput:size(3))

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

            local logits_t = self.decoder:forward(
                {encoderOutput_b, decoderInput_t})
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
            local cell_t = self.decoder.module:get(1):get(2):get(2).cells[step]
            local output_t = self.decoder.module:get(1):get(2):get(2).outputs[step]

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

