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
    local segmentsTable = {}
    local max_encoder_length = 0
    local max_decoder_length = 0
    local max_segments_length = 0

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
        local input, output, segsStr = table.unpack(
            stringx.split(line, " ||| "))

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
       
        local segments = {}
        table.insert(segments, 1)
        for s, segStr in ipairs(stringx.split(segsStr, " ")) do
            table.insert(segments, tonumber(segStr) + 1)
        end
        table.insert(segments, #inputIds + 2)
        table.insert(segmentsTable, segments)


        if #inputIds > max_encoder_length then
           max_encoder_length = #inputIds
        end
        if #outputIds > max_decoder_length then
           max_decoder_length = #outputIds
        end
        if #segments > max_segments_length then
           max_segments_length = #segments
        end

    end

    local datasetSize = #encoderTable
    local encoderSize = max_encoder_length + 1
    local decoderSize = max_decoder_length + 1
    local segments = torch.FloatTensor(datasetSize, max_segments_length):zero()

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
        segments[i]:narrow(1, 1, #segmentsTable[i]):copy(
            torch.FloatTensor(segmentsTable[i]):add(offset - 1))

    end

    return vocab, ids, encoderInput, decoderInput, decoderOutput, segments
end

function data.tostring(datum, segments, ids)
    if datum:dim() == 1 then
        local T = {}
        for s=1,segments:size(1) - 1 do 
            local t = {}
            if segments[s+1] == 0 then break end
            local chunk = datum:narrow(
                1, segments[s], segments[s+1] - segments[s])
            for i=1,chunk:size(1) do 
                if chunk[i] > 3 then
                    table.insert(t, ids[chunk[i]])
                end
            end
            table.insert(T, table.concat(t, " "))

        end
        return table.concat(T, "   ")

    elseif datum:dim() == 2 then
        local D = {}
        
        for k=1,datum:size(1) do
            local datum_k = datum[k]
            local segment_k = segments[k]
            
            local T = {}
            for s=1,segment_k:size(1) - 1 do 
                local t = {}
                if segment_k[s+1] == 0 then break end
                local chunk = datum_k:narrow(
                    1, segment_k[s], segment_k[s+1] - segment_k[s])
                for i=1,chunk:size(1) do 
                    if chunk[i] > 3 then
                        table.insert(t, ids[chunk[i]])
                    end
                end
                table.insert(T, table.concat(t, " "))

            end
            table.insert(D, table.concat(T, "   "))
        end
           
        return D

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

return data
