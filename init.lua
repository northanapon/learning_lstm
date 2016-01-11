require 'nn'
require 'nngraph'

lstm = {}

lstm.utils = {}
-- An important function to make recursive/recurrent neural network work.
-- Copy (same pointer) all parameters from sharing unit to target unit
function lstm.utils.share_parameters(sharing, target)
--    if sharing.parameters then
--        local s_params, s_grad_params = sharing:parameters()
--        local t_params, t_grad_params = target:parameters()
--        for i = 1, #s_params do
--            t_params[i]:set(s_params[i])
--            t_grad_params[i]:set(s_grad_params[i])
--        end
--    else
--        error('no parameters to share')
--    end
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