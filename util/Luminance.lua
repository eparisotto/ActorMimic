
require "nn"
require "image"

local lumin = torch.class('nn.Luminance', 'nn.Module')

function lumin:__init()
end

function lumin:forward(x)
	local x = x
	if x:dim() > 3 then
		x = x[1]
	end

	x = image.rgb2y(x)
	return x
end

function lumin:updateOutput(input)
	return self:forward(input)
end

function lumin:float()
end
