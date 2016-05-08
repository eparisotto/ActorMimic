
require 'networks.convnet'

return function(args)
    args.n_units        = {32, 64, 64}
    args.filter_size    = {8, 4, 3}
    args.filter_stride  = {4, 2, 1}
	 args.padding        = {1, 0, 0}
    args.n_hid          = {512}
    args.nl             = nn.ReLU

    return create_network(args)
end

