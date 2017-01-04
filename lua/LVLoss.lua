local LVLoss, parent = torch.class('nn.LVLoss', 'nn.Criterion')

function LVLoss:__init()
   parent.__init(self)
   self.criterion = nn.ClassNLLCriterion()
   self.criterion.sizeAverage = false
   self.criterion = nn.MaskZeroCriterion(self.criterion, 1)
   if torch.isTypeOf(criterion, 'nn.ModuleCriterion') then
      error("SequencerCriterion shouldn't decorate a ModuleCriterion. "..
         "Instead, try the other way around : "..
         "ModuleCriterion decorates a SequencerCriterion. "..
         "Its modules can also be similarly decorated with a Sequencer.")
   end

   self.outputs = torch.Tensor()
   self.gradyll = torch.Tensor()
   self.gradlvll = torch.Tensor()
   self.lvl = torch.Tensor()
   self.clones = {}
   self.gradInput = {}
end

function LVLoss:getStepCriterion(step)
   assert(step, "expecting step at arg 1")
   local criterion = self.clones[step]
   if not criterion then
      criterion = self.criterion:clone()
      self.clones[step] = criterion
   end
   return criterion
end

function LVLoss:forward(lvll, yll, target)

   self.output = 0
   local nStep = yll:size(1)
   local lvl = self.lvl:typeAs(lvll):resizeAs(lvll)
   torch.exp(lvl, lvll)

   local outputs = self.outputs:resize(nStep)
   for i=1,nStep do
      local criterion = self:getStepCriterion(i)
      outputs[i] = criterion:forward(yll[i], target[i])
   end
    self.output = outputs:sum()

   return self.output

end

function LVLoss:backward(lvll, yll, target)

    local gradlvll = self.gradlvll:typeAs(lvll):resizeAs(lvll)
    local gradyll = self.gradyll:typeAs(yll):resizeAs(yll)
    local nStep = yll:size(1)

    for i=1,nStep do 
        local criterion = self:getStepCriterion(i)
        gradyll[i]:copy(criterion:backward(yll[i], target[i]))
        gradyll[i]:mul(self.lvl[i])
    end
    gradlvll:copy(self.lvl):cmul(self.outputs)
    return gradlvll, gradyll
end
