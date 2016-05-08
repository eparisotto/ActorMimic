--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

if not zoo then
    require "util.initenv"
end

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Train Agent in Environment:')
cmd:text()
cmd:text('Options:')

cmd:option('-framework', '', 'name of training framework')
cmd:option('-envs', '', 'name of environment to use in array {env1,env2,...}')
cmd:option('-game_path', '', 'path to environment file (ROM)')
cmd:option('-env_params', '', 'string of environment parameters')
cmd:option('-pool_frms', '',
           'string of frame pooling parameters (e.g.: size=2,type="max")')
cmd:option('-actrep', 1, 'how many times to repeat action')
cmd:option('-random_starts', 0, 'play action 0 between 1 and random_starts ' ..
           'number of times at the start of each training episode')

cmd:option('-name', '', 'filename used for saving network and training history')
cmd:option('-network', '', 'reload pretrained network')
cmd:option('-agent', '', 'name of agent file to use')
cmd:option('-agent_params', '', 'string of agent parameters')
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-saveNetworkParams', false,
           'saves the agent network in a separate file')
cmd:option('-saveTransitions', false,
			  'saves the transition tables of the network')
cmd:option('-prog_freq', 5*10^3, 'frequency of progress output')
cmd:option('-save_freq', 5*10^4, 'the model is saved every save_freq steps')
cmd:option('-eval_freq', 10^4, 'frequency of greedy evaluation')
cmd:option('-save_versions', 0, '')
cmd:option('-draw', 0, 'whether to display each frame.')
cmd:option('-drawsave', 0, 'whether to save each frame to a file.')
cmd:option('-drawsave_path', '', 'the path to save the frames to')

cmd:option('-steps', 10^5, 'number of training steps to perform')
cmd:option('-eval_steps', 10^5, 'number of evaluation steps')

cmd:option('-verbose', 2,
           'the higher the level, the more information is printed to screen')
cmd:option('-threads', 1, 'number of BLAS threads')
cmd:option('-gpu', -1, 'gpu flag')
cmd:option('-cudnn', 0, 'whether to use cuDNN kernels')
cmd:option('-cudnn_fastest', 0, 'whether to set fastest flag in cuDNN')

cmd:text()

local opt = cmd:parse(arg)

--- General setup.
local game_envs, game_actions, agent, opt = setup(opt)

-- override print to always flush the output
local old_print = print
local print = function(...)
    old_print(...)
    io.flush()
end

local learn_start = agent.learn_start or math.inf
local start_time = sys.clock()

local windows = {}

local reward_counts = {}
local episode_counts = {}
local time_history = {}
local v_history = {}
local qmax_history = {}
local td_history = {}
local reward_history = {}
local step = 0
time_history[1] = 0

local sum_total_reward = 0
local total_rewards    = {}
local nrewards         = {}
local nepisodes        = {}
local episode_rewards  = {}

local screens = {}
local rewards = {}
local terminals = {}

for i=1,#game_envs do
   total_rewards[i]   = 0
   nrewards[i]        = 0
   nepisodes[i]       = 0
   episode_rewards[i] = 0

   screens[i], rewards[i], terminals[i] = game_envs[i]:getState()
end


print("Iteration ..", step)
while step < opt.steps do
   step = step + 1

   local action_indices = agent:perceive(rewards, screens, terminals)
	agent:updateAgent()
   
   for i=1,#game_envs do
      local action_index = action_indices[i]
      
      if opt.draw ~= 0 then
			windows[i] = image.display({image=screens[i], win=windows[i]})
      end
      if opt.drawsave ~= 0 then
			image.save(opt.drawsave_path .. opt.name .. opt.envs[i] ..
							  '-f' .. step .. '.png',
						  screens[i]:squeeze())
      end
      
      -- game over? get next game!
      if not terminals[i] then
			screens[i], rewards[i], terminals[i] = game_envs[i]:step(game_actions[action_index], true)
      else
			if opt.random_starts > 0 then
				screens[i], rewards[i], terminals[i] = game_envs[i]:nextRandomGame()
			else
				screens[i], rewards[i], terminals[i] = game_envs[i]:newGame()
			end
      end
   end
   
   if step % opt.prog_freq == 0 then
      assert(step==agent.numSteps, 'trainer step: ' .. step ..
					 ' & agent.numSteps: ' .. agent.numSteps)
      print("Steps: ", step)
      agent:report()
      collectgarbage()
   end
   
   if step%1000 == 0 then collectgarbage() end
   
   if step % opt.eval_freq == 0 and step > learn_start then
      
      for i=1,#game_envs do
			
			screens[i], rewards[i], terminals[i] = game_envs[i]:newGame()
			
			nrewards[i] = 0
			nepisodes[i] = 0
			episode_rewards[i] = 0
			total_rewards[i] = 0
      end
      
      local eval_time = sys.clock()
      for estep=1,opt.eval_steps do
			local action_indices = agent:perceive(rewards, screens, terminals, true, 0.05)
			
			for i=1,#game_envs do
				local action_index = action_indices[i]
				
				if opt.draw ~= 0 then
					windows[i] = image.display({image=screens[i], win=windows[i]})
				end
				if opt.drawsave ~= 0 then
					image.save(opt.drawsave_path .. opt.name .. opt.envs[i] ..
									  '-f' .. step .. '-eval' .. estep .. '.png',
								  screens[i]:squeeze())
				end
				
				-- Play game in test mode (episodes don't end when losing a life)
				screens[i], rewards[i], terminals[i] = game_envs[i]:step(game_actions[action_index])
	    
				-- record every reward
				episode_rewards[i] = episode_rewards[i] + rewards[i]
				if rewards[i] ~= 0 then
					nrewards[i] = nrewards[i] + 1
				end
				
				if terminals[i] then
					total_rewards[i] = total_rewards[i] + episode_rewards[i]
					episode_rewards[i] = 0
					nepisodes[i] = nepisodes[i] + 1
					screens[i], rewards[i], terminals[i] = game_envs[i]:nextRandomGame()
				end
			end
			
			if estep%1000 == 0 then collectgarbage() end
      end
      
      -- Make sure we don't output 0 reward just because we didn't finish an episode
      for i=1,#game_envs do
			if nepisodes[i] == 0 then
				total_rewards[i] = episode_rewards[i]
			end
      end
      
      sum_total_reward = torch.Tensor(total_rewards):sum()
      
      eval_time = sys.clock() - eval_time
      start_time = start_time + eval_time
      local ind = #reward_history+1
      sum_total_reward = sum_total_reward / math.max(1, torch.Tensor(nepisodes):sum())
      
      if #reward_history == 0 or sum_total_reward > torch.Tensor(reward_history):max() then
			agent.best_network = agent.network:clone()
      end
      
      agent:compute_validation_statistics()
      agent:print_validation_statistics()
      
      reward_history[ind] = sum_total_reward
      reward_counts[ind]  = torch.Tensor(nrewards):sum()
      episode_counts[ind] = torch.Tensor(nepisodes):sum()

      time_history[ind+1] = sys.clock() - start_time
      local time_dif      = time_history[ind+1] - time_history[ind]
      local training_rate = opt.actrep*opt.eval_freq/time_dif
      
      outstr = string.format(
			'\nSteps: %d (frames: %d), sum total reward: %.2f, epsilon: %.2f, lr: %G, ' ..
				'training time: %ds, training rate: %dfps, testing time: %ds, ' ..
				'testing rate: %dfps, num. ep.: %d, num. rewards: %d',
			step, step*opt.actrep, sum_total_reward, agent.ep, agent.lr, time_dif,
			training_rate, eval_time, opt.actrep*opt.eval_steps/eval_time,
			torch.Tensor(nepisodes):sum(), torch.Tensor(nrewards):sum())
      for i=1,#game_envs do
			total_rewards[i] = total_rewards[i] / math.max(1, nepisodes[i])
			outstr = outstr .. '\n\tGame ' .. opt.envs[i] .. ': '
			outstr = outstr .. string.format('reward: %.2f, #ep: %d, #rewards: %d',
														total_rewards[i], nepisodes[i], nrewards[i])
      end
      print(outstr)
	end
   
   if step % opt.save_freq == 0 or step == opt.steps then
      --
      local s, a, r, s2, term = agent.valid_s, agent.valid_a, agent.valid_r,
             agent.valid_s2, agent.valid_term
      agent.valid_s, agent.valid_a, agent.valid_r, agent.valid_s2,
             agent.valid_term = nil, nil, nil, nil, nil, nil, nil
      local optim = agent.optim
      agent.optim = nil
		local transtab = agent.transitions
		if not opt.saveTransitions then
			agent.transitions = nil
		end
	
      local filename = opt.name
      if opt.save_versions > 0 then
			filename = filename .. "_" .. math.floor(step / opt.save_versions)
      end
      filename = filename
      torch.save(filename .. ".t7", {agent = agent,
				     model = agent.network,
				     best_model = agent.best_network,
				     reward_history = reward_history,
				     reward_counts = reward_counts,
				     episode_counts = episode_counts,
				     time_history = time_history,
				     v_history = v_history,
				     td_history = td_history,
				     qmax_history = qmax_history,
				     arguments=opt})
      if opt.saveNetworkParams then
			local nets = {network=w:clone():float()}
			torch.save(filename..'.params.t7', nets, 'ascii')
      end
      agent.valid_s, agent.valid_a, agent.valid_r, agent.valid_s2,
             agent.valid_term = s, a, r, s2, term
      agent.optim = optim
		agent.transitions = transtab
		print('Saved:', filename .. '.t7')
      io.flush()
      collectgarbage()
   end
end
