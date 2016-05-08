
require 'networks.convnet'

return function(args)
   args.n_units       = {256, 512, 512, 512}
   args.filter_size   = {8, 4, 3, 3}
   args.filter_stride = {4, 2, 1, 1}
	args.padding       = {1, 0, 0, 0}
   args.n_hid         = {2048, 1024}
   args.nl            = nn.ReLU

   return create_network(args)
end
