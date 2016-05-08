--[[ Gradient Scale Layer.
Forward pass is the identity while backward pass scales the incoming gradients
by a provided factor.
--]]

local GradientScale, parent = torch.class('nn.GradientScale', 'nn.Module')

function GradientScale:__init(g_scale)
   parent.__init(self)
   self.g_scale = g_scale or 1
   self.train   = true -- if false, lets gradients pass through
end

function GradientScale:updateOutput(input)
   self.output = input
   return self.output
end

function GradientScale:updateGradInput(input, gradOutput)
   if self.train then
      -- If train is true, scale the incoming gradient from the top layer
      self.gradInput:resizeAs(gradOutput)
      self.gradInput:copy(gradOutput)
      self.gradInput:mul(self.g_scale)
   else
      -- If train is false, be an backward layer
      self.gradInput = gradOutput
   end
   return self.gradInput
end

function GradientScale:__tostring__()
   local s = string.format('%s(%.2f)', torch.type(self), self.g_scale)
   return s
end
