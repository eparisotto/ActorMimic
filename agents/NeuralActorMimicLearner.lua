
if not zoo then
   require 'initenv'
end

local aml, nl = torch.class('zoo.NeuralActorMimicLearner', 'zoo.NeuralLearner')

function aml:__init(args)
	
   -- Actor-Mimic parameters
   self.objective     = "ce"                      -- cross-entropy or l2.
   self.temperature   = args.temperature or 1.0   -- temperature of softmax if obj=CE.
   self.actor         = args.actor or "student"   -- sample from 'student' or 'expert'.
   self.featreg       = args.featreg or false     -- whether to use feature regression.
   self.featreg_scale = args.featreg_scale or 1.0 -- scaling factor for feature regression.
	
   nl.__init(self, args)
	
   -- Create Actor-Mimic minibatch+validation structures
   self.full_q  = torch.FloatTensor(self.minibatch_size*self.n_games, self.n_actions)
   self.valid_q = torch.FloatTensor(self.valid_size*self.n_games, self.n_actions)
   if self.featreg then
      -- Build feature guidance storage
      self.full_f          = {}
      self.valid_f         = {}
      self.featreg_targets = {}
      for i=1,self.n_games do
         local n_featlayer = #self.expertnet[i].modules-1
         local featsize    = self.expertnet[i].modules[n_featlayer].output:size(2)
         self.full_f[i]  = torch.FloatTensor(self.minibatch_size, featsize)
         self.valid_f[i] = torch.FloatTensor(self.valid_size,     featsize)
         self.featreg_targets[i] = nil
         if self.gpu >= 0 then
            self.featreg_targets[i] = torch.CudaTensor(self.minibatch_size*self.n_games, featsize)
         else
            self.featreg_targets[i] = torch.FloatTensor(self.minibatch_size*self.n_games, featsize)
         end
      end
   end
	
   -- Softmax function
   self.SoftMaxCE = nn.SoftMax()   
	
   -- Set objective function
	io.write('Objective:')
   if objective == 'l2' then
		print(' L2')
      self.getUpdate = self.getUpdateL2
   else
		print(' CE')
      self.getUpdate = self.getUpdateCE
   end
end

function aml:createNetwork(args)
   nl.createNetwork(self, args)
	
   -- Load expert networks
   self.expertnet = {}
   if not (type(args.expertnet_prefix) == 'string') then
      error('The expert folder is not a string')
   end
   for i=1,self.n_games do
      local netname = args.expertnet_prefix .. self.game_names[i] .. '.t7'
      collectgarbage()
      print('Creating Expert Network ' .. i .. ' from ' .. netname)
      local msg, exp = pcall(torch.load, netname)
      if not msg then
         error('Error loading expert network')
      end
      self.expertnet[i] = exp.model
		self.expertnet[i]:cuda()
		self.expertnet[i]:forward(torch.zeros(1,unpack(self.input_dims)):cuda())
		
      local nl = #self.expertnet[i].modules
      local  l =  self.expertnet[i].modules[nl]
		
      -- Either the expert has the full 18-action output or the game-specific valid actions
      if (l.output:size(2) ~= self.n_actions) and (l.output:size(2) ~= #self.game_actions[i]) then
			error('Expert network has wrong number of actions as output')
      end
		
      -- If the expert network has only game-specific valid actions as output, convert
      -- it into a full 18-action output
      if l.output:size(2) == #self.game_actions[i] then
         local newl = nn.Linear(l.weight:size(2), self.n_actions)
         newl.weight:zero()
         newl.bias:zero()
			
         for j=1,#self.game_actions[i] do
            newl.weight[{{self.game_actions[i][j]},{}}]:copy(l.weight[{{j},{}}])
            newl.bias[self.game_actions[i][j]] = l.bias[j]
         end
         newl:cuda()
			
         -- Sanity check
         local valcount = 1
         for j=1,self.n_actions do
            if j == self.game_actions[i][valcount] then
               assert(torch.sum(torch.abs(torch.add(l.weight[valcount], -1, newl.weight[j]))) == 0)
               assert(math.abs(l.bias[valcount] - newl.bias[j]) == 0)
               valcount = valcount + 1
            else
               assert(torch.sum(torch.abs(newl.weight[j])) == 0)
               assert(newl.bias[j] == 0)
            end
         end
         self.expertnet[i].modules[nl] = newl
      end
		
      if self.gpu and self.gpu >= 0 then
         self.expertnet[i]:cuda()
      else
         self.expertnet[i]:float()
      end
		if self.cudnn and self.cudnn > 0 then
			cudnn.convert(self.expertnet[i], cudnn)
		end
      exp = nil
		print(self.expertnet[i])
   end
	
   -- augment the network with the prediction networks
   if self.featreg then
      self.featregnet = args.featregnet
		
      print('Creating Feature Regression Network from ' .. self.featregnet)
      local msg, err = pcall(require, self.featregnet)
      if not msg then
         print(err)
         error('error loading feature regression network')
      end
      self.featregnet = err
      self.featregnet = self:featregnet()
		
      -- Augment the current network with the feature regression networks
      self.network.modules[#self.network.modules] = self.featregnet
      print(self.network)
		
      if self.gpu and self.gpu >= 0 then
         self.network:cuda()
      else
         self.network:float()
      end
   end
end

function aml:getUpdateL2(args)
   local outputs, targets, mask
   outputs = args.outputs
   targets = args.targets
   mask    = args.mask
	
   if self.gpu >= 0 then
      outputs = outputs:cuda()
      targets = targets:cuda()
   end
   targets:add(-1, outputs:cmul(mask))
	
   return targets
end

function aml:getUpdateCE(args)
   local outputs, targets, masks
   outputs = args.outputs:float()
   targets = args.targets
   mask    = args.mask
	
   -- number of samples / game
   local g_size = outputs:size(1) / self.n_games 
	
   for i=1,self.n_games do
      local idx = {{(i-1)*g_size+1, i*g_size}, {}}
      local subtargets = targets[idx]:maskedSelect(mask[idx])
      local suboutputs = outputs[idx]:maskedSelect(mask[idx])
      subtargets:resize(g_size, #self.game_actions[i])
      suboutputs:resize(g_size, #self.game_actions[i])
		
      -- Divide target net outputs by temperature before softmax
      subtargets:div(self.temperature)
		
      subtargets:copy(self.SoftMaxCE:forward(subtargets))
			:add(-1, self.SoftMaxCE:forward(suboutputs))
      targets[idx]:zero():maskedCopy(mask[idx], subtargets)      
   end
   if self.gpu >= 0 then targets = targets:cuda() end
	
   return targets
end

function aml:getUpdateFeatureL2(args)
   local outputs, targets
   outputs = args.outputs
   targets = args.targets
   if self.gpu >= 0 then
      outputs = outputs:cuda()
      targets = targets:cuda()
   end
	
   local gradtargets = targets:clone():add(-1, outputs):mul(self.featreg_scale)
   return gradtargets
end

function aml:calcTargets()
	
   -- Zero out the expert Q-value storage
   self.full_q:zero()
   
   -- Run through each game and query the expert for guidance
   for i=1,self.n_games do
      local idx = {{(i-1)*self.minibatch_size+1, i*self.minibatch_size},{}}
		
      local expertout = self.expertnet[i]:forward(self.full_s[idx]):float()
      self.full_q[idx]:copy(expertout)
		
      -- save the features, if needed
      if self.featreg then
			self.full_f[i]
				:copy(self.expertnet[i].modules[#self.expertnet[i].modules-1].output:clone():float())
      end
   end
   -- Mask out invalid actions
   self.full_q:maskedFill(torch.ne(self.mb_mask, 1), 0)
	
   -- Forward pass through the mimic
   -- The output here will be a table if featreg is on
   local outputs = self.network:forward(self.full_s)
	
   -- Take just the action outputs
   local actout = outputs
   if self.featreg then actout = outputs[1] end
	
   -- Calculate the action targets
   local acttar = self:getUpdate{outputs=actout, targets=self.full_q, mask=self.mb_mask}
	
   -- Are we done?
   if not self.featreg then
      return acttar
   end
	
   -- Build full target table including action targets
   local fulltargets = {acttar}
   
   -- Calculate feature regression targets
   for i=1,self.n_games do
      -- zero out the parts of the targets that correspond to different games
      fulltargets[i+1] = self.featreg_targets[i]:zero()
		
      -- Fill in the targets for the correct game
      local idx = {{(i-1)*self.minibatch_size+1,i*self.minibatch_size},{}}
      fulltargets[i+1][idx]
			:copy(self:getUpdateFeatureL2{outputs=outputs[i+1][idx], targets=self.full_f[i]})
   end
	
   return fulltargets
end

function aml:forwardNetwork(gameidx, state)
	-- If we are sampling from student network
	if self.actor == 'student' then
		if self.featreg then
			return self.network:forward(state)[1]:float():squeeze()
		end
		return self.network:forward(state):float():squeeze()
	end
	-- otherwise, sample from expert
	return self.expertnet[gameidx]:forward(state):float():squeeze()
end

function aml:compute_validation_statistics()
   -- Convert the states to gpu, if necessary
   if self.gpu >= 0 then
      self.valid_s  = self.valid_s:cuda()
      self.valid_s2 = self.valid_s2:cuda()
   end
	
   local outputs = self.network:forward(self.valid_s)
	
   local actout = outputs
   if self.featreg then actout = outputs[1] end
	
   local targets = actout:clone():zero()
	
   local update = self:getUpdate{outputs=actout, targets=self.valid_q, mask=self.val_mask}
	
   self.action_loss = {}
   for i=1,self.n_games do
      self.action_loss[i] = update[{{(i-1)*self.valid_size+1,i*self.valid_size},{}}]:abs():mean()
   end
	
   if not self.featreg then
      return nil
   end
   
   self.featreg_loss = {}
   for i=1,self.n_games do
      local idx = {{(i-1)*self.valid_size+1,i*self.valid_size}}
      self.featreg_loss[i] = self:getUpdateFeatureL2{outputs=outputs[i+1][idx], targets=self.valid_f[i]}
      self.featreg_loss[i] = self.featreg_loss[i]:abs():mean()
   end
	
   -- Reconvert back to CPU RAM to save memory
   if self.gpu >= 0 then
      self.valid_s  = self.valid_s:float()
      self.valid_s2 = self.valid_s2:float()
   end
end

function aml:print_validation_statistics()
   print('Per-game validation statistics:')
   for i=1,self.n_games do
      io.write('\t' .. self.game_names[i] .. ': ' .. self.objective, self.action_loss[i])
      if self.featreg then
			print(', featreg', self.featreg_loss[i])
      else
			print('')
      end
   end
end
