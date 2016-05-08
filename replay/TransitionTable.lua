
require 'image'

local trans = torch.class('zoo.TransitionTable')


function trans:__init(args)
	self.stateDim = args.stateDim
	self.numActions = args.numActions
	self.histLen = args.histLen
	self.maxSize = args.maxSize or 1024^2
	self.histType = args.histType or "linear"
	self.histSpacing = args.histSpacing or 1
	self.zeroFrames = args.zeroFrames or 1
	self.gpu = args.gpu
	self.numEntries = 0
	self.insertIndex = 0
	
	self.histIndices = {}
	local histLen = self.histLen
	if self.histType == "linear" then
		-- History is the last histLen frames.
		self.recentMemSize = self.histSpacing*histLen
		for i=1,histLen do
			self.histIndices[i] = i*self.histSpacing
		end
	elseif self.histType == "exp2" then
        -- The ith history frame is from 2^(i-1) frames ago.
		self.recentMemSize = 2^(histLen-1)
		self.histIndices[1] = 1
		for i=1,histLen-1 do
			self.histIndices[i+1] = self.histIndices[i] + 2^(7-i)
		end
	elseif self.histType == "exp1.25" then
		-- The ith history frame is from 1.25^(i-1) frames ago.
		self.histIndices[histLen] = 1
		for i=histLen-1,1,-1 do
			self.histIndices[i] = math.ceil(1.25*self.histIndices[i+1])+1
		end
		self.recentMemSize = self.histIndices[1]
		for i=1,histLen do
			self.histIndices[i] = self.recentMemSize - self.histIndices[i] + 1
		end
	end
	
	self.s = torch.ByteTensor(self.maxSize, self.stateDim):fill(0)
	self.a = torch.LongTensor(self.maxSize):fill(0)
	self.r = torch.zeros(self.maxSize)
	self.t = torch.ByteTensor(self.maxSize):fill(0)
	
	-- Tables for storing the last histLen states.  They are used for
	-- constructing the most recent agent state more easily.
	self.recent_s = {}
	self.recent_a = {}
	self.recent_t = {}

	self:loadBuffers(args.bufferSize)
end

function trans:loadBuffers(bufferSize)

	self.bufferSize = bufferSize or 1024

	local s_size = self.stateDim*self.histLen
	self.buf_s      = torch.ByteTensor(self.bufferSize, s_size):fill(0)
	self.buf_a      = torch.LongTensor(self.bufferSize):fill(0)
	self.buf_r      = torch.zeros(self.bufferSize)
	self.buf_s2     = torch.ByteTensor(self.bufferSize, s_size):fill(0)
	self.buf_a2     = torch.LongTensor(self.bufferSize):fill(0)
	self.buf_term   = torch.ByteTensor(self.bufferSize):fill(0)
	
	if self.gpu and self.gpu >= 0 then
		self.gpu_s  = self.buf_s:float():cuda()
		self.gpu_s2 = self.buf_s2:float():cuda()
	end
	
	-- Set this so that we refill the buffer
	self.buf_ind = self.bufferSize + 1
end


function trans:reset()
	self.numEntries = 0
	self.insertIndex = 0
end


function trans:size()
	return self.numEntries
end


function trans:empty()
	return self.numEntries == 0
end


function trans:fill_buffer()
   assert(self.numEntries >= self.bufferSize)
   -- clear CPU buffers
   self.buf_ind = 1
   for buf_ind=1,self.bufferSize do
      local idx = self:sample_one(buf_ind)
      
      local s, a, r, s2, a2, term = self:get(idx)
      self.buf_s[buf_ind]:copy(s)
      self.buf_a[buf_ind] = a
      self.buf_r[buf_ind] = r
      self.buf_s2[buf_ind]:copy(s2)
      self.buf_a2[buf_ind] = a2
      self.buf_term[buf_ind] = term
   end
   self.buf_s  = self.buf_s:float():div(255)
   self.buf_s2 = self.buf_s2:float():div(255)
   if self.gpu and self.gpu >= 0 then
      self.gpu_s:copy(self.buf_s)
      self.gpu_s2:copy(self.buf_s2)
   end
end


function trans:sample_one(buf_ind)
	assert(self.numEntries > 1)
	local index
	local valid = false
	while not valid do
		-- start at 2 because of previous action
		index = torch.random(2, self.numEntries-self.recentMemSize) -- -1 for a2 (on-policy)
		if self.t[index+self.recentMemSize-1] == 0 then
			valid = true
		end
	end
	
	return index
end


function trans:sample(batch_size)
   local batch_size = batch_size or 1
   assert(batch_size <= self.bufferSize)
	
   if not self.buf_ind or self.buf_ind + batch_size - 1 > self.bufferSize then
      self:fill_buffer()
   end
   
   local index = self.buf_ind
   
   self.buf_ind = self.buf_ind+batch_size
   local range = {{index, index+batch_size-1}}
   
   local buf_s, buf_a, buf_r, buf_s2, buf_a2, buf_term = self.buf_s,
        self.buf_a, self.buf_r, self.buf_s2, self.buf_a2, self.buf_term
   if self.gpu and self.gpu >=0  then
        buf_s = self.gpu_s
        buf_s2 = self.gpu_s2
   end

   return buf_s[range], buf_a[range], buf_r[range], buf_s2[range],
          buf_a2[range], buf_term[range]
end


function trans:concatFrames(index, use_recent)
	if use_recent then
		s, t = self.recent_s, self.recent_t
	else
		s, t = self.s, self.t
	end
	
	local fullstate = s[1].new()
	fullstate:resize(self.histLen, unpack(s[1]:size():totable()))
	
	-- Zero out frames from all but the most recent episode.
	local zero_out = false
	local episode_start = self.histLen
	
	for i=self.histLen-1,1,-1 do
		if not zero_out then
			for j=index+self.histIndices[i]-1,index+self.histIndices[i+1]-2 do
				if t[j] == 1 then
					zero_out = true
					break
				end
			end
		end
		
		if zero_out then
			fullstate[i]:zero()
		else
			episode_start = i
		end
	end
	
	if self.zeroFrames == 0 then
		episode_start = 1
	end
	
	-- Copy frames from the current episode.
	for i=episode_start,self.histLen do
		fullstate[i]:copy(s[index+self.histIndices[i]-1])
	end
	
	return fullstate
end

function trans:get_recent()
	-- Assumes that the most recent state has been added, but the action has not
	return self:concatFrames(1, true):float():div(255)
end


function trans:get(index)
	local s = self:concatFrames(index)
	local s2 = self:concatFrames(index+1)
	local ar_index = index+self.recentMemSize-1
	
	return s, self.a[ar_index], self.r[ar_index], s2, self.a[ar_index+1], self.t[ar_index+1]
end


function trans:add(s, a, r, term)
	assert(s, 'State cannot be nil')
	assert(a, 'Action cannot be nil')
	assert(r, 'Reward cannot be nil')
	
	-- Increment until at full capacity
	if self.numEntries < self.maxSize then
		self.numEntries = self.numEntries + 1
	end

	-- Always insert at next index, then wrap around
	self.insertIndex = self.insertIndex + 1
	-- Overwrite oldest experience once at capacity
	if self.insertIndex > self.maxSize then
		self.insertIndex = 1
	end
	
	-- Overwrite (s,a,r,t) at insertIndex
	self.s[self.insertIndex] = s:clone():float():mul(255)
	self.a[self.insertIndex] = a
	self.r[self.insertIndex] = r
	if term then
		self.t[self.insertIndex] = 1
	else
		self.t[self.insertIndex] = 0
	end
end

function trans:updatePriority(indices, priority)
end

function trans:add_recent_state(s, term)
	local s = s:clone():float():mul(255):byte()
	if #self.recent_s == 0 then
		for i=1,self.recentMemSize do
			table.insert(self.recent_s, s:clone():zero())
			table.insert(self.recent_t, 1)
		end
	end
	
	table.insert(self.recent_s, s)
	if term then
		table.insert(self.recent_t, 1)
	else
		table.insert(self.recent_t, 0)
	end
	
	-- Keep recentMemSize states.
	if #self.recent_s > self.recentMemSize then
		table.remove(self.recent_s, 1)
		table.remove(self.recent_t, 1)
	end
end


function trans:add_recent_action(a)
	if #self.recent_a == 0 then
		for i=1,self.recentMemSize do
			table.insert(self.recent_a, 1)
		end
	end
	
	table.insert(self.recent_a, a)
	
	-- Keep recentMemSize steps.
	if #self.recent_a > self.recentMemSize then
		table.remove(self.recent_a, 1)
	end
end
