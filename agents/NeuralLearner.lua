
if not zoo then
   require 'initenv'
end

local nl, agent = torch.class('zoo.NeuralLearner', 'zoo.Agent')


function nl:__init(args)
   agent.__init(self, args)
   
   --- epsilon annealing
   self.ep_start   = args.ep_start or 1      -- Epsilon at the first frame.
   self.ep         = self.ep_start           -- Exploration probability.
   self.ep_end     = args.ep_end or self.ep  -- Epsilon at the end of annealing.
   self.ep_endt    = args.ep_endt or 1000000 -- Number of steps before annealing ends.
   
   ---- learning rate annealing
   self.lr_start       = args.lr or 0.01          -- Learning rate at start of learning.
   self.lr             = self.lr_start            -- Learning rate.
   self.lr_end         = args.lr_end or self.lr   -- Learning rate at end of annealing.
   self.lr_endt        = args.lr_endt or 1000000  -- Number of steps before annealing ends.
   self.wc             = args.wc or 0             -- L2 weight cost.
   self.minibatch_size = args.minibatch_size or 1 -- Size of a minibatch (per game)
   self.valid_size     = args.valid_size or 500   -- Size of validation set (per game)
	self.clip_grads     = args.clip_grads or 0     -- max norm of gradients (0 for no clipping)

   --- Learning parameters
   self.discount       = args.discount or 0.99 -- Discount factor.
   self.update_freq    = args.update_freq or 1 -- How often to update parameters.
   self.n_replay       = args.n_replay or 1    -- Number of points to replay per learning step.
   self.learn_start    = args.learn_start or 0 -- Number of steps after which learning starts.

   -- Create the network
   self:createNetwork(args)   
   if self.gpu and self.gpu >= 0 then
      self.network:cuda()
      self.tensor_type = torch.CudaTensor
   else
      self.network:float()
      self.tensor_type = torch.FloatTensor
   end

	-- Create the optimizer
	self:createOptimizer(args)

	-- Initialize general agent statistics
	self:initializeStats(args)

   -- Initialize validation statistics
	self:initializeValidationStats(args)

	-- Create minibatch + validation structures
	self:createBatchStructures(args)

	-- Create masks for the different games
	self:createMasks(args)
end

function nl:createNetwork(args)
   self.network = args.network
   assert(self.network ~= nil)

   -- check whether there is a network file
   if not (type(self.network) == 'string') then
      error("Must provide a string as network to createNetwork")
   end

   local msg, err = pcall(require, self.network)
   if not msg then
      -- try to load saved agent
      local err_msg, exp = pcall(torch.load, self.network)
      if not err_msg then
			error("Could not find network file " .. self.network)
      end

      -- Do we load best network or the last network?
      self.best = args.best
      if self.best and exp.best_model then
			self.network = exp.best_model
      else
			self.network = exp.model
      end

		-- If the last layer is a ConcatTable, extract just the action output
		-- <!> fix loading/saving with network modifications
		local outnum = #self.network.modules
		if torch.type(self.network.modules[outnum]) == 'nn.ConcatTable' then
			self.network.modules[outnum] = self.network.modules[outnum].modules[1].modules[1]
		end

		exp = nil
		collectgarbage()

		print(self.network)
   else
      print('Creating Agent Network from ' .. self.network)
      self.network = err
      self.network = self:network()
   end
	
   -- Are some parameters frozen?
   self.n_freeze_layers = args.n_freeze_layers or 0
   assert(self.n_freeze_layers < #self.network.modules)
   if self.n_freeze_layers > 0 then -- freeze first 'self.n_freeze_layers' layers

      -- Add a ScaleGradient layer with scalar 0 to stop gradients from flowing
      -- into the frozen layers
      self.network:insert(nn.GradientScale(0.0), self.n_freeze_layers+1)
      
      print('Updated frozen network:')
      print(self.network)   
   end

	if self.gpu and self.gpu >= 0 then
		self.network:cuda()
	else
		self.network:float()
	end
end

function nl:createOptimizer(args)
   -- Store all network parameters
   self.w, self.dw = self.network:getParameters()
   self.dw:zero()

	-- Create optimizer
	self.optimizer_name = args.optim
	self.optim = zoo[args.optim](args, self.dw)
end

function nl:initializeValidationStats(args)
	-- Initialize validation statistics
	self.v_avg     = 0 -- V running average.
	self.tderr_avg = 0 -- TD error running average.
end

function nl:initializeStats(args)
	-- Initialize reward rescaling variables
	self.o_max = {}
	self.r_max = {}
	for i=1,self.n_games do
		self.r_max[i] = 1 -- Used to divide the reward during Q-learning, initialize at 1.
		self.o_max[i] = 1 -- Used to track the maximum action output seen so far.
	end
end

function nl:createBatchStructures(args)
	-- Create minibatch structures
	self.full_s   = nil
	self.full_a   = torch.LongTensor(self.minibatch_size*self.n_games):fill(0)
	self.full_r   = torch.zeros(self.minibatch_size*self.n_games)
	self.full_s2  = nil
	self.full_a2  = torch.LongTensor(self.minibatch_size*self.n_games):fill(0)
	self.full_t   = torch.ByteTensor(self.minibatch_size*self.n_games):fill(0)

	if self.gpu >= 0 then
		self.full_s = torch.CudaTensor(self.minibatch_size*self.n_games, self.state_dim*self.hist_len)
	else
		self.full_s = torch.ByteTensor(self.minibatch_size*self.n_games, self.state_dim*self.hist_len)
	end
	self.full_s2 = self.full_s:clone()

	-- Create validation structures
	-- Don't keep the states continuously in GPU memory, since this can cause memory to be filled
	-- especially if there isn't much GPU RAM
	self.valid_s  = torch.ByteTensor(self.valid_size*self.n_games, self.state_dim*self.hist_len)
	self.valid_a  = torch.LongTensor(self.valid_size*self.n_games):fill(0)
	self.valid_r  = torch.zeros(self.valid_size*self.n_games):fill(0)
	self.valid_s2 = self.valid_s:clone()
	self.valid_a2 = torch.LongTensor(self.valid_size*self.n_games):fill(0)
	self.valid_t  = torch.ByteTensor(self.valid_size*self.n_games):fill(0)
end

function nl:createMasks(args)
	-- Create the game action and minibatch masks
	self.mb_mask  = torch.ByteTensor(self.n_games*self.minibatch_size, self.n_actions):zero()
	self.val_mask = torch.ByteTensor(self.n_games*self.valid_size,     self.n_actions):zero()
	for i=1,self.n_games do
		local midx = {(i-1)*self.minibatch_size+1, i*self.minibatch_size}
		local vidx = {(i-1)*self.valid_size+1,     i*self.valid_size}

		for j=1,#self.game_actions[i] do
			self.mb_mask[{midx,self.game_actions[i][j]}]  = 1
			self.val_mask[{vidx,self.game_actions[i][j]}] = 1
		end
	end
end

function nl:getUpdate(args)
   -- override this function in derived class
   -- NOTE: self.network:forward() MUST be called in the derived class implementation of
   -- this function for gradients to be correct
   assert(false)
end

function nl:loadMinibatchSamples()
   for i=1,self.n_games do
      assert(self.transitions[i]:size() > self.minibatch_size)
      local s, a, r, s2, a2, term = self.transitions[i]:sample(self.minibatch_size)

      -- Combine the states into a large full-state matrix
      local sind = (i-1)*self.minibatch_size + 1
      local eind = i*self.minibatch_size
      self.full_s[{{sind, eind},{}}]:copy(s)
      self.full_a[{{sind, eind}}]:copy(a)
      self.full_r[{{sind, eind}}]:copy(r)
      self.full_s2[{{sind, eind},{}}]:copy(s2)
      self.full_a2[{{sind, eind}}]:copy(a2)
      self.full_t[{{sind, eind}}]:copy(term)
   end
end

function nl:calcTargets()
   local targets = self:getUpdate{s=self.full_s, a=self.full_a, r=self.full_r, 
											 s2=self.full_s2, a2=self.full_a2, term=self.full_t,
											 mask=self.mb_mask}   
   return targets
end

function nl:learnMinibatch()
   self:loadMinibatchSamples()
   local targets = self:calcTargets()

   -- zero gradients of parameters
   self.dw:zero()

   -- get new gradient
   self.network:backward(self.full_s, targets)

   -- add weight cost to gradient
   self.dw:add(-self.wc, self.w)

	-- Clip gradients, if requested
	if self.clip_grads > 0 then
		local gradnorm = torch.norm(self.dw)
		if gradnorm > self.clip_grads then
			self.dw:mul(self.clip_grads/gradnorm)
		end
	end

   -- compute linearly annealed learning rate
   local t = math.max(0, self.numSteps - self.learn_start)
   self.lr = (self.lr_start - self.lr_end) * (self.lr_endt - t)/self.lr_endt + self.lr_end
   self.lr = math.max(self.lr, self.lr_end)

   -- Optimize
   self.w, self.dw = self.optim:optimize(self.lr, self.w, self.dw)
end

function nl:preperceive(rewards, rawstates, terminals, testing, testing_ep)
   if self.numSteps == self.learn_start+1 and not testing then
      self:sample_validation_data()
   end
end

function nl:chooseAction(gameidx, state, testing_ep)   
   return self:eGreedy(gameidx, state, testing_ep)
end

function nl:postperceive(states, actions, rewards, rawstates, terminals, testing, testing_ep)
end

function nl:updateAgent()
   -- Do some learning updates
   if self.numSteps > self.learn_start and self.numSteps % self.update_freq == 0 then
      for i=1,self.n_replay do
			self:learnMinibatch()
      end
   end

   -- Updates the training counter
   agent.updateAgent(self)
end

function nl:eGreedy(gameidx, state, testing_ep)
   self.ep = testing_ep or (self.ep_end +
			       math.max(0, (self.ep_start - self.ep_end) * (self.ep_endt -
			       math.max(0, self.numSteps - self.learn_start))/self.ep_endt))

   -- Epsilon greedy
   if torch.uniform() < self.ep then
      local act = torch.random(1, #self.game_actions[gameidx])
      return self.game_actions[gameidx][act]
   else
      return self:greedy(gameidx, state)
   end
end

function nl:forwardNetwork(gameidx, state)
   return self.network:forward(state):t():float():squeeze()
end

function nl:greedy(gameidx, state)
   -- Turn single state into minibatch. Needed for convolutional nets
   if state:dim() == 2 then
      assert(false, 'input must be at least 3D')
      state = state:resize(1, state:size(1), state:size(2))
   end

   if self.gpu >= 0 then
      state = state:cuda()
   end

   local o = self:forwardNetwork(gameidx, state)
   local maxo, besta

   maxo  = o[self.game_actions[gameidx][1]]
   besta = { self.game_actions[gameidx][1] }
   -- Evaluate all other game-specific actions (with random tie-breaking)
   for a=2,#self.game_actions[gameidx] do
      local act = self.game_actions[gameidx][a]
      if o[act] > maxo then
	 besta = { act }
	 maxo = o[act]
      elseif o[act] == maxo then
	 besta[#besta+1] = act
      end
   end

   if maxo > self.o_max[gameidx] then
      self.o_max[gameidx] = maxo
   end

   local r = torch.random(1, #besta)
   return besta[r]
end

function nl:report()
   print(get_weight_norms(self.network))
   print(get_grad_norms(self.network))
end

function nl:sample_validation_data()
   -- samples some states for validation reporting
   for i=1,self.n_games do
      assert(self.transitions[i]:size() > self.valid_size)
      local s, a, r, s2, a2, t = self.transitions[i]:sample(self.valid_size)

      -- Combine the states into a large full state matrix
      local sind = (i-1)*self.valid_size+1
      local eind =     i*self.valid_size
      self.valid_s[ {{sind,eind}}]:copy(s)
      self.valid_a[ {{sind,eind}}]:copy(a)
      self.valid_r[ {{sind,eind}}]:copy(r)
      self.valid_s2[{{sind,eind}}]:copy(s2)
      self.valid_a2[{{sind,eind}}]:copy(a2)
      self.valid_t[ {{sind,eind}}]:copy(t)
   end
end

function nl:compute_validation_statistics()
   -- must override in derived class
end

function nl:print_validation_statistics()
   -- must overrite in derived class
end
