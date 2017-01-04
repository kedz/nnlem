local BatchIndex, Parent = torch.class('nn.BatchIndex', 'nn.Module')

function BatchIndex:__init()
    self.output = torch.Tensor()
end

function BatchIndex:updateOutput(input)

    local t = input[1]
    local index = input[2]

    local output = self.output:resize(t:size(1), 1):typeAs(t)

    for batch=1,t:size(1) do
        local index_b = index[batch][1]
        if index_b > 0 then
            output[batch][1] = t[batch][index_b]
        else
            output[batch][1] = 0
        end
    end
   
    return output

end
