
zoo = {}

require 'torch'
require 'nn'
require 'nngraph'
require 'image'

require 'util.nnutils'
require 'util.Rectifier'
require 'util.Scale'
require 'util.Luminance'
require 'util.GradientScale'

require 'agents.Agent'
require 'agents.NeuralLearner'
require 'agents.NeuralQLearner'
require 'agents.NeuralActorMimicLearner'

require 'replay.TransitionTable'

require 'optimizers.rmsprop'
require 'optimizers.adam'

function preprocOptions(_opt)
	--preprocess options:
	--- convert options strings to tables
	_opt.pool_frms = str_to_table(_opt.pool_frms)
	_opt.env_params = str_to_table(_opt.env_params)
	_opt.agent_params = str_to_table(_opt.agent_params)
	_opt.envs = str_to_table(_opt.envs)
	if _opt.agent_params.transition_params then
		_opt.agent_params.transition_params = str_to_table(_opt.agent_params.transition_params)
	end
	return _opt
end


function torchSetup(_opt)
   _opt = _opt or {}
   local opt = table.copy(_opt)
   assert(opt)
   
   -- preprocess options:
   --- convert options strings to tables
   if opt.pool_frms then
      opt.pool_frms = str_to_table(opt.pool_frms)
   end
   if opt.env_params then
      opt.env_params = str_to_table(opt.env_params)
   end
   if opt.agent_params then
      opt.agent_params = str_to_table(opt.agent_params)
      opt.agent_params.gpu       = opt.gpu
		opt.agent_params.cudnn     = opt.cudnn
      opt.agent_params.best      = opt.best
      opt.agent_params.verbose   = opt.verbose
      if opt.network ~= '' then
			opt.agent_params.network = opt.network
      end
   end
   
   --- general setup
   opt.tensorType =  opt.tensorType or 'torch.FloatTensor'
   torch.setdefaulttensortype(opt.tensorType)
   if not opt.threads then
      opt.threads = 4
   end
   torch.setnumthreads(opt.threads)
   if not opt.verbose then
      opt.verbose = 10
   end
   if opt.verbose >= 1 then
      print('Torch Threads:', torch.getnumthreads())
   end
   
   --- set gpu device
   if opt.gpu and opt.gpu >= 0 then
      require 'cutorch'
      require 'cunn'
      if opt.gpu == 0 then
			local gpu_id = tonumber(os.getenv('GPU_ID'))
			if gpu_id then opt.gpu = gpu_id+1 end
      end
      if opt.gpu > 0 then cutorch.setDevice(opt.gpu) end
      opt.gpu = cutorch.getDevice()
      print('Using GPU device id:', opt.gpu-1)

		if opt.cudnn and opt.cudnn > 0 then
			require 'cudnn'
			if opt.cudnn_fastest then
				-- pull out all the stops
				cudnn.fastest = true
			end
		end
	else
		opt.gpu = -1
		if opt.verbose >= 1 then
			print('Using CPU code only. GPU device id:', opt.gpu)
		end
   end
   
   --- set up random number generators
   -- removing lua RNG; seeding torch RNG with opt.seed and setting cutorch
   -- RNG seed to the first uniform random int32 from the previous RNG;
   -- this is preferred because using the same seed for both generators
   -- may introduce correlations; we assume that both torch RNGs ensure
   -- adequate dispersion for different seeds.
   math.random = nil
   opt.seed = opt.seed or 1
   torch.manualSeed(opt.seed)
   if opt.verbose >= 1 then
      print('Torch Seed:', torch.initialSeed())
   end
   local firstRandInt = torch.random()
   if opt.gpu >= 0 then
      cutorch.manualSeed(firstRandInt)
      if opt.verbose >= 1 then
			print('CUTorch Seed:', cutorch.initialSeed())
      end
   end
   
   return opt
end

function setupEnvs(opt)
	 -- load training framework and environment
   local framework = require(opt.framework)
   assert(framework)

   print('loading games:')
   for k,v in pairs(opt.envs) do
      print(k, v)
   end

	local gameEnvs = {}
   for i=1,#opt.envs do
      -- Set the environment for this specific game
      opt.env = opt.envs[i]
      gameEnvs[i] = framework.GameEnvironment(opt)
   end

	return opt, gameEnvs
end

function setupAgent(_opt, gameActions, envNames, envActions)
	-- agent options
   _opt.agent_params.name    = _opt.name
   _opt.agent_params.actions = gameActions
   _opt.agent_params.gpu     = _opt.gpu
	_opt.agent_params.cudnn   = _opt.cudnn
   _opt.agent_params.best    = _opt.best
   _opt.agent_params.n_games = #envNames
   _opt.agent_params.verbose = _opt.verbose

   -- Store the game-specific actions
   _opt.agent_params.game_names   = envNames
   _opt.agent_params.game_actions = envActions
   for i=1,#envNames do
      -- Set to index-1 instead of index-0
      for j=1,#_opt.agent_params.game_actions[i] do
         _opt.agent_params.game_actions[i][j] = _opt.agent_params.game_actions[i][j] + 1
      end
   end

   local agent = zoo[_opt.agent](_opt.agent_params)

	return _opt, agent
end


function setup(_opt)
   assert(_opt)
   
   -- preprocess options
	_opt = preprocOptions(_opt)
   
   -- first things first
   local opt = torchSetup(_opt)
   
   -- load training framework and environments
	local gameEnvs
	opt, gameEnvs = setupEnvs(opt)

	-- Build total game actions
	local gameActions = {}
	-- Set all actions as legal, since this simplifies handling multiple games
	for i=1,18 do
		gameActions[i] = i-1
	end

	-- Build the game-specific actions
   envActions = {}
   for i=1,#opt.envs do
      envActions[i] = gameEnvs[i]:getActions()
   end

	-- Load the agent
	_opt, agent = setupAgent(_opt, gameActions, opt.envs, envActions)
   
   if opt.verbose >= 1 then
      print('Set up Torch using these options:')
      for k, v in pairs(opt) do
			print(k, v)
      end
   end
   
   return gameEnvs, gameActions, agent, opt
end

--- other functions
function str_to_table(str)
   if type(str) == 'table' then
      return str
   end
   if not str or type(str) ~= 'string' then
      if type(str) == 'table' then
			return str
      end
      return {}
   end
   local ttr
   if str ~= '' then
      local ttx=tt
      loadstring('tt = {' .. str .. '}')()
      ttr = tt
      tt = ttx
   else
      ttr = {}
   end
   return ttr
end

function table.copy(t)
   if t == nil then return nil end
   local nt = {}
   for k, v in pairs(t) do
      if type(v) == 'table' then
			nt[k] = table.copy(v)
      else
			nt[k] = v
      end
   end
   setmetatable(nt, table.copy(getmetatable(t)))
   return nt
end
