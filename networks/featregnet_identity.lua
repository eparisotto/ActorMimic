--[[
--]]

require 'networks.featregnet'

return function(args)
   args.featreg_stopgrad = false
   args.featreg_identity = true
	args.final_nl         = false

   return create_featreg_network(args)
end
