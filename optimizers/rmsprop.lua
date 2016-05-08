
if not zoo then
   require 'util.initenv'
end

local rmsprop = torch.class('zoo.RMSpropOptimizer')

function rmsprop:__init(args, dw)
   self.beta1 = args.beta1
   self.beta2 = args.beta2
   self.gamma = args.gamma

   self.deltas = dw:clone():fill(0)
   self.tmp    = dw:clone():fill(0)
   self.g      = dw:clone():fill(0)
   self.g2     = dw:clone():fill(0)
end

function rmsprop:optimize(lr, w, dw)
   -- g2 = beta2 * g2 + (1-beta2) * dw^2
   self.tmp:cmul(dw, dw)
   self.g2:mul(self.beta2):add(1-self.beta2, self.tmp)

   -- g = beta1 * g + (1-beta1) * dw
   self.g:mul(self.beta1):add(1-self.beta1, dw)

   -- tmp = sqrt(g2 - g^2 + gamma)
   self.tmp:cmul(self.g, self.g)
   self.tmp:mul(-1)
   self.tmp:add(self.g2)
   self.tmp:add(self.gamma)
   self.tmp:sqrt()

   -- delta_w = lr * dw / sqrt(g^2 + g2 + gamma)
   self.deltas:mul(0):addcdiv(lr, dw, self.tmp)

   -- update weights
   w:add(self.deltas)

   return w, dw
end

