
if not zoo then
   require 'initenv'
end

local agent = torch.class('zoo.Agent')

function agent:__init(args)

	-- Store the original input arguments
	self.args = args

   -- Name of the agent
   self.name = args.name

   -- How much debug output to print.
   self.verbose = args.verbose

   -- Game parameters
   self.state_dim    = args.state_dim    -- Dimension of each (non-concatenated) frame
   self.actions      = args.actions      -- The actions we can choose
   self.n_actions    = #self.actions     -- The number of actions possible
   self.n_games      = args.n_games      -- The number of games we are playing simultaneously
   self.game_actions = args.game_actions -- The number of actions in a specific game
   self.game_names   = args.game_names   -- The name of each game

   -- State representation parameters
   self.hist_len       = args.hist_len or 1 -- The number of frames to concatenate together
   self.rescale_r      = args.rescale_r     -- Whether to rescale the rewards when used as targets
   self.max_reward     = args.max_reward    -- The max reward we allow (clipped at this value)
   self.min_reward     = args.min_reward    -- The min reward we allo (clipped at this value0
   self.ncols          = args.ncols or 1    -- number of color channels in input
   self.input_dims     = args.input_dims or {self.hist_len*self.ncols, 84, 84} -- state dimension
   self.histType       = args.histType or "linear"  -- history type to use
   self.histSpacing    = args.histSpacing or 1      -- spacing between consecutive frames in state

   -- Transition table parameters
   self.replay_memory = args.replay_memory or 1000000 -- size of the transition table
   self.bufferSize    = args.bufferSize or 512        -- size of the transition table buffer

   -- Whether to use GPU, if needed
   self.gpu = args.gpu

	-- Whether to use cuDNN, if needed
	self.cudnn = args.cudnn

   -- Number of training steps so far.
   self.numSteps = 0

   -- Load preprocessing network.
   self.preproc = args.preproc -- name of preprocessing network
   if not (type(self.preproc == 'string')) then
      error('The preprocessing is not a string')
   end
   msg, err = pcall(require, self.preproc)
   if not msg then
      error("Error loading preprocessing net " .. self.preproc)
   end
   self.preproc = err
   self.preproc = self:preproc()
   self.preproc:float()

   -- Initialize per-game temporary storage
   self.transitions = {}
   self.lastStates = {}
   self.lastActions = {}
   self.lastTerminals = {}
	for i=1,self.n_games do
		self.lastStates[i]    = nil
		self.lastActions[i]   = nil
		self.lastTerminals[i] = nil
	end

	-- Load in the transition tables
	self.replay_table = args.replay_table
	if not (type(self.replay_table) == 'string') then
		error("Must provide a string location to the replay table")
	end

	for i=1,self.n_games do
		-- Create transition table.
		---- assuming the transition table always gets floating point input
		---- (Float or Cuda tensors) and always returns one of the two, as required
		---- internally it always uses ByteTensors for states, scaling and
		---- converting accordingly
		local transition_args = {
			stateDim = self.state_dim, numActions = self.n_actions,
			histLen = self.hist_len, gpu = self.gpu,
			maxSize = self.replay_memory,
			histType = self.histType, histSpacing = self.histSpacing,
			bufferSize = self.bufferSize,
			alpha = self.rt_alpha, beta_start = self.rt_beta_start, beta_endt = self.rt_beta_endt,
			recalcPartition = self.rt_recalcPartition
		}
		
		self.transitions[i] = zoo[self.replay_table](transition_args)
	end
end

function agent:preprocess(rawstate)
   if not self.preproc then
      return rawstate
   end
   return self.preproc:forward(rawstate:float()):clone():reshape(self.state_dim)
end

function agent:preperceive(rewards, rawstates, terminals, testing, testing_ep)
   return
end

function agent:chooseAction(gameidx, state, testing_ep)
   -- Must be overwritten
   assert(false)
end

function agent:postperceive(states, actions, rewards, rawstates, terminals, testing, testing_ep)
   return
end

function agent:updateAgent()
   self.numSteps = self.numSteps + 1
end

function agent:perceive(rewards, rawstates, terminals, testing, testing_ep)

   self:preperceive(rewards, rawstates, terminals, testing, testing_ep)

   local curStates     = {}
   local actionIndices = {}
   for i=1,self.n_games do
      if self.max_reward then
			rewards[i] = math.min(rewards[i], self.max_reward)
      end
      if self.min_reward then
			rewards[i] = math.max(rewards[i], self.min_reward)
      end
      if self.rescale_r then
			self.r_max[i] = math.max(self.r_max[i], rewards[i])
      end
		
      -- Preprocess state (will be set to nil if terminal
      local state = self:preprocess(rawstates[i]):float()
      self.transitions[i]:add_recent_state(state, terminals[i])
		
      curStates[i] = self.transitions[i]:get_recent()
      curStates[i] = curStates[i]:resize(1, unpack(self.input_dims))

      -- Store transition s, a, r, s2, a2, t
      if self.lastStates[i] and not testing then
			self.transitions[i]:add(self.lastStates[i], self.lastActions[i],
											rewards[i], self.lastTerminals[i])
      end
		
      -- Select action
      local actionIndex = 1
      if not terminals[i] then
			actionIndex = self:chooseAction(i, curStates[i], testing_ep)
      end
		
      self.transitions[i]:add_recent_action(actionIndex)
		
      self.lastStates[i]    = state:clone()
      self.lastActions[i]   = actionIndex
      self.lastTerminals[i] = terminals[i]
		
      if not terminals[i] then
			actionIndices[i] = actionIndex
      else
			actionIndices[i] = 0
      end
   end

   self:postperceive(curStates, actionIndices, rewards,
							rawstates, terminals, testing, testing_ep)
   
   return actionIndices
end

function agent:report()
   assert(false)
end
