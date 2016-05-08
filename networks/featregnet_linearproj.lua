--[[
--]]

require 'networks.featregnet'

return function(args)
   args.n_hid = {}
   args.nl    = nn.ReLU

   args.featreg_stopgrad = false
   args.featreg_identity = false
	args.final_nl         = false

   return create_featreg_network(args)
end
