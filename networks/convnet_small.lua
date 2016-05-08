--[[
--]]

require 'networks.convnet'

return function(args)
	args.n_units       = {32, 32}
	args.filter_size   = {16, 8}
	args.filter_stride = {8, 4}
	args.padding       = {1, 0}
	args.n_hid         = {512}
	args.nl            = nn.ReLU

	return create_network(args)
end
