data = {}

function data.readVocab(path)
    local vocab = {}
    local ids = {}

    local vid = 0
    for line in io.lines(path) do
        vid = vid + 1
        vocab[line] = vid
        table.insert(ids, line)
    end

    return vocab, ids
end

function data.read(path, vocab, ids, showProgress)
    local vid
    if vocab == nil or ids == nil then
        vocab = {}
        ids = {}
        vocab["<se>"] = 1
        vocab["<sd>"] = 2
        vocab["<ed>"] = 3
        ids[1] = "<se>"
        ids[2] = "<sd>"
        ids[3] = "<ed>"
        vid = 4
    end

    local encoderTable = {}
    local decoderTable = {}
    local max_encoder_length = 0
    local max_decoder_length = 0

    -- Get total number of lines in the file
    local cmd = assert(io.popen("wc -l < " .. path))
    local totalExamples = tonumber(cmd:read())
    cmd:close()

    local exampleIndex = 1
    for line in io.lines(path) do
        if showProgress then
            xlua.progress(exampleIndex, totalExamples)
        end
        exampleIndex = exampleIndex + 1
        --if num % 100000 == 0 then
        --    print(num)
        --end
        local input, output = table.unpack(stringx.split(line, " ||| "))
        local inputTokens = stringx.split(input, " ")
        local inputIds = {}
        for i=1,#inputTokens do
            local token = inputTokens[i]
            if not vocab[token] then
                vocab[token] = vid
                ids[vid] = token
                vid = vid + 1
            end
            table.insert(inputIds, vocab[token])
        end
        table.insert(encoderTable, inputIds)

        local outputTokens = stringx.split(output, " ")
        local outputIds = {}
        for i=1,#outputTokens do
            local token = outputTokens[i]
            if not vocab[token] then
                vocab[token] = vid
                ids[vid] = token
                vid = vid + 1
            end
            table.insert(outputIds, vocab[token])
        end
        table.insert(decoderTable, outputIds)

        if #inputIds > max_encoder_length then
           max_encoder_length = #inputIds
        end
        if #outputIds > max_decoder_length then
           max_decoder_length = #outputIds
        end

    end

    local datasetSize = #encoderTable
    local encoderSize = max_encoder_length + 1
    local decoderSize = max_decoder_length + 1

    local encoderInput = torch.FloatTensor(datasetSize, encoderSize):zero()
    local decoderInput = torch.FloatTensor(datasetSize, decoderSize):zero()
    local decoderOutput = torch.FloatTensor(datasetSize, decoderSize):zero()
    for i=1,datasetSize do
        local offset = encoderSize - #encoderTable[i]
        encInp_i = encoderInput[i]:narrow(1, offset, #encoderTable[i] + 1)
        encInp_i[1] = 1
        for j=2,#encoderTable[i] + 1 do
            encInp_i[j] = encoderTable[i][j-1]
        end

        decInp_i = decoderInput[i]:narrow(1, 1, #decoderTable[i] + 1)
        decOut_i = decoderOutput[i]:narrow(1, 1, #decoderTable[i] + 1)
        decInp_i[1] = 2
        for j=2,#decoderTable[i] + 1 do
            decInp_i[j] = decoderTable[i][j-1]
        end
        decOut_i:narrow(1, 1, #decoderTable[i]):copy(
            decInp_i:narrow(1, 2, #decoderTable[i]))
        decOut_i[#decoderTable[i] + 1] = 3
    end

    return vocab, ids, encoderInput, decoderInput, decoderOutput
end

function data.tostring(datum, ids)
    if datum:dim() == 1 then
        local t = {}
        for s=1,datum:size(1) do
            if datum[s] > 3 then
                table.insert(t, ids[datum[s]])
            end
        end
        return table.concat(t, " ")

    elseif datum:dim() == 2 then
        T = {}
        for i=1,datum:size(1) do
            local datum_i = datum[i]
            local t = {}
            for s=1,datum_i:size(1) do
                if datum_i[s] > 3 then
                    table.insert(t, ids[datum_i[s]])
                end
            end
            table.insert(T, table.concat(t, " "))
        end
        return T

    else
        print("Arg 1 must be vector or matrix.")
        os.exit()
    end

    os.exit()
end

function data:batchIter(encIn, decIn, decOut, batchSize)
    self.encIn_buf = data.encIn_buf or torch.Tensor():typeAs(encIn)
    self.encIn_buf:resize(batchSize, encIn:size(2))
    self.decIn_buf = data.decIn_buf or torch.Tensor():typeAs(decIn)
    self.decIn_buf:resize(batchSize, decIn:size(2))
    self.decOut_buf = data.decOut_buf or torch.Tensor():typeAs(decOut)
    self.decOut_buf:resize(batchSize, decOut:size(2))
    self.I_buf = torch.LongTensor()

    local I = torch.randperm(self.I_buf, encIn:size(1))
    local i = 1
    local t = 0

    local maxSteps = math.floor((encIn:size(1) + batchSize) / batchSize)


    local function iter()
        t = t + 1
        if i > encIn:size(1) then return nil end

        local hi = math.min(i+batchSize-1, encIn:size(1))
        local step = hi - i + 1
        self.encIn_buf:index(encIn, 1, I:narrow(1,i,step))
        self.decIn_buf:index(decIn, 1, I:narrow(1,i,step))
        self.decOut_buf:index(decOut, 1, I:narrow(1,i,step))
        i = i + step

        return {encIn=self.encIn_buf, decIn=self.decIn_buf,
                decOut=self.decOut_buf, t=t, maxSteps=maxSteps}

    end

    return iter

end

