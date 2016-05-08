
require "util.initenv"

-- args here will be "self" of calling class
function create_featreg_network(args)
   local nl = #args.network.modules

   local rndexp = torch.randn(32,4,84,84)
   if args.gpu >= 0 then
      rndexp = rndexp:cuda()
   end

   -- Run through an example to get output sizes
   args.network:forward(rndexp)
   for i=1,args.n_games do
      args.expertnet[i]:forward(rndexp)
   end

   local net = nn.ConcatTable()
   print(args.network.modules)
   local n_acts = args.network.modules[nl-1].output:size(1)
   local infeat = args.network.modules[nl-1].output:size(2)

   -- Add the previous output, but use a stop gradient layer if we don't want the previous
   -- objectives to influence the features
   local netcolumn = nn.Sequential()
   if args.featreg_stopgrad then
      netcolumn:add(nn.StopGradient())
   end
   netcolumn:add(args.network.modules[nl])
   net:add(netcolumn)

   -- Add the feature regression columns for each game 
   local totalout = n_acts
   for i=1,args.n_games do
      local expnl = #args.expertnet[i].modules
      local outfeat   = args.expertnet[i].modules[expnl-1].output:size(2)
      local netcolumn = nn.Sequential()

      if args.featreg_identity then
			-- If we want the feature layer of the multitask network to match exactly the feature layer of the
			-- experts, we use an identity transform for the column
			netcolumn:add(nn.Identity())
      else
			-- Hidden layers
			local lowfeat = infeat
			for j=1,#args.n_hid do
				netcolumn:add(nn.Linear(lowfeat, args.n_hid[i]))
				netcolumn:add(args.nl())
				lowfeat = args.n_hid[i]
			end
			-- Add the last regression layer
			netcolumn:add(nn.Linear(lowfeat, outfeat))
			if args.final_nl then
				netcolumn:add(args.nl()) 
			end
      end
      totalout = totalout + outfeat
		
      -- Add the regression column to the network output
      net:add(netcolumn)
   end

   return net
end
