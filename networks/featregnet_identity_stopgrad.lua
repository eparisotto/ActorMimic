--[[
--]]

require 'networks.featregnet'

return function(args)
   args.featreg_stopgrad = true
   args.featreg_identity = true
	args.final_nl         = false

   return create_featreg_network(args)
end
