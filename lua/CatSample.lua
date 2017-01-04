require 'randomkit'

local CatSample = torch.class('CatSample')

CatSample.sampleBuffer = torch.Tensor()
CatSample.indexBuffer = torch.LongTensor()

function CatSample.batchSample(X, P)

    for batch=1,X:size(1) do
        CatSample.sample(X[batch], P[batch])
    end
    return X
end

function CatSample.sample(x, p)
    -- shamelessly stolen from wikipedia. Thanks internet!
    
    local n = x:size(1)
    local r = 1
    local s = 1
    local sampleBuffer = CatSample.sampleBuffer:resize(n):typeAs(x)

    for i=1,p:size(1) do
        local v = randomkit.binomial(n, p[i] / r)
        for j=1,v do
            sampleBuffer[s] = i
            s = s + 1
        end
        n = n - v
        if n == 0 then break end
        r = r - p[i]
    end

    local I = torch.randperm(CatSample.indexBuffer, x:size(1))
    x:indexCopy(1, I, sampleBuffer)
    return x
end

