
if not zoo then
    require 'initenv'
end

local nql, nl = torch.class('zoo.NeuralQLearner', 'zoo.NeuralLearner')


function nql:__init(args)
   nl.__init(self, args)

   -- Q-learning parameters
   self.clip_delta     = args.clip_delta     -- Whether to clip the temporal difference error.
   self.target_q       = args.target_q or 0  -- Whether to use a target network. (0 for none)

   -- Q-learning validation statistics
   self.v_avg     = {} -- Per-game V running average
   self.tderr_avg = {} -- Per-game TD error running average
   for i=1,self.n_games do
      self.v_avg[i]     = 0
      self.tderr_avg[i] = 0
   end

   -- Load target network, if needed
   if self.target_q > 0 then
      self.target_network = self.network:clone()
   end
end

function nql:getNetworkTarget(args)
	local s2   = args.s2
   local mask = args.mask

   local target_q_net
   if self.target_q > 0 then
      target_q_net = self.target_network
   else
      target_q_net = self.network
   end

   -- Compute Q-learning next-step action value:
   -- max_a Q(s_2, a).
   local q2_full = target_q_net:forward(s2):float()

   -- Since we're masking out values, we don't want to select target action values that are not
   -- valid, so we fill all non-valid action-values with the minimum of the entire batch output
   -- This will ensure that a non-valid action value will not be chosen as a target incorrectly
   local fillVal = torch.min(q2_full)
   q2_full:maskedFill(torch.ne(mask,1), fillVal)
   self.q2_max = q2_full:max(2)

   return self.q2_max
end

function nql:getUpdate(args)
    local s, a, r, term
    local q, q2, q2_target
    local termnot

    s = args.s
    a = args.a
    r = args.r
    term = args.term

    -- calculate (1-terminal)
    termnot = term:clone():float():mul(-1):add(1)

    -- Compute the next-step action-value target
    q2_target = self:getNetworkTarget(args)

    -- Compute q2 = (1-terminal) * gamma * Q_target(s2,a') for some a' depending on method
    q2 = q2_target:clone():mul(self.discount):cmul(termnot)

    -- delta = r + (1-terminal) * gamma * max_a Q(s2, a) - Q(s, a)
    self.delta = r:clone():float()
    if self.rescale_r then
       local rangesize = q2:size(1) / self.n_games
       for i=1,self.n_games do
	  local g_range = {{(i-1)*rangesize+1,i*rangesize}}
	  self.delta[g_range]:div(self.r_max[i])
       end
    end
    self.delta:add(q2)

    -- q = Q(s,a)
    local q_all = self.network:forward(s):float()
    q = torch.FloatTensor(q_all:size(1))
    for i=1,q_all:size(1) do
        q[i] = q_all[i][a[i]]
    end
    self.delta:add(-1, q)

    if self.clip_delta then
        self.delta[self.delta:ge(self.clip_delta)] = self.clip_delta
        self.delta[self.delta:le(-self.clip_delta)] = -self.clip_delta
    end

    local targets = q_all:clone():zero()
    for i=1,math.min(targets:size(1), a:size(1)) do
        targets[i][a[i]] = self.delta[i]
    end

    if self.gpu >= 0 then targets = targets:cuda() end

    return targets
end

function nql:perceive(rewards, rawstates, terminals, testing, testing_ep)
   actionIndices = nl.perceive(self, rewards, rawstates, terminals, testing, testing_ep)

   -- Have a chance to update the target q network every
   -- 'target_q' number of iterations
   if self.target_q > 0 and self.numSteps % self.target_q == 1 then
      self.target_network = nil
      collectgarbage()
      self.target_network = self.network:clone()
   end

   return actionIndices
end

function nql:compute_validation_statistics()
   --[[
   -- Convert the states to gpu, if necessary
   if self.gpu >= 0 then
      self.valid_s  = self.valid_s:cuda()
      self.valid_s2 = self.valid_s2:cuda()
   end
   
   local targets = self:getUpdate{s=self.valid_s, a=self.valid_a, r=self.valid_r, 
				   s2=self.valid_s2, a2=self.valid_a2, term=self.valid_t,
				   mask=self.val_mask}
   
   for i=1,self.n_games do
      g_range = {{(i-1)*self.valid_size+1, i*self.valid_size}}
      self.v_avg[i]     = self.q2_max[g_range]:mean()
      self.tderr_avg[i] = self.delta[ g_range]:clone():abs():mean()
   end
   
   -- Reconvert back to CPU RAM to save memory
   if self.gpu >= 0 then
      self.valid_s  = self.valid_s:float()
      self.valid_s2 = self.valid_s2:float()
   end
   --]]
end


function nql:print_validation_statistics()
   --[[
   print('Per-game validation statistics:')
   for i=1,self.n_games do
      print("\t" .. self.game_names[i] ..": V", self.v_avg[i], 
	    "TD error", self.tderr_avg[i], "Qmax", self.o_max[i])
   end
   --]]

end
