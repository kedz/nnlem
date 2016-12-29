local stopToken = 3

eval = {}

function eval.coarseAccuracy(predictedOutput, expectedOutput)

    assert(predictedOutput:size(1) == expectedOutput:size(1))
    local batchSize = predictedOutput:size(1)
    local sequenceLength = math.min(predictedOutput:size(2),
                                    expectedOutput:size(2))

    local numCorrect = 0



    for batch=1,batchSize do
        local predOutput_i = predictedOutput[batch]
        local expOutput_i = expectedOutput[batch]
        local correct = true
        for step=1,sequenceLength do
            if predOutput_i[step] ~= expOutput_i[step] then
                correct = false
                break
            end
            if predOutput_i[step] == stopToken then
                break
            end
        end
        if correct == true then
            numCorrect = numCorrect + 1
        end
    end

    return numCorrect / batchSize

end

return eval

