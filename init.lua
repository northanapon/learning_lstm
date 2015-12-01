require 'nn'
require 'nngraph'

lstm = {}

lstm.utils = {}
-- An important function to make recursive/recurrent neural network work.
-- Copy (same pointer) all parameters from sharing unit to target unit
function lstm.utils.share_parameters(sharing, target)
    if torch.isTypeOf(target, 'nn.gModule') then
        for i = 1, #target.forwardnodes do
            local node = target.forwardnodes[i]
            if node.data.module then
                lstm.utils.share_parameters(
                    sharing.forwardnodes[i].data.module,
                    node.data.module)
            end
        end
    elseif torch.isTypeOf(target, 'nn.Module') then
        target:share(sharing, 'weight', 'bias', 'gradWeight', 'gradBias')
    else
        error('cannot share parameters of the argument type')
    end
end

include('LSTM.lua')