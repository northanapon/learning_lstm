-- Adapted from https://github.com/karpathy/char-rnn
-- A set of functions to create and run LSTM network

local LSTM = {}

function LSTM.lstm(config)
    local input_dim = config.input_dim
    local hidden_dim = config.hidden_dim
    local num_layers = config.num_layers
    local emb_unit = config.emb_unit
    local output_unit = config.output_unit
    local input_idx = nn.Identity()():annotate{name='input index'}
    local prev_C = nn.Identity()():annotate{name='prev C'}
    local prev_H = nn.Identity()():annotate{name='prev H'}
    local inputs = {input_idx, prev_C, prev_H}
    local c, h, outputs = {}, {}, {}
    for l = 1, num_layers do
        local prev_c = nn.SelectTable(l)(prev_C):annotate{name='prev c at ' .. l}
        local prev_h = nn.SelectTable(l)(prev_H):annotate{name='prev h at ' .. l}
        local dim, input
        if l == 1 then
            input = emb_unit(inputs[1]):annotate{name='embedding'}
            dim = input_dim
        else
            input = h[l-1]
            dim = hidden_dim
        end
        local i2h = nn.Linear(dim, 4 * hidden_dim)(input):annotate{name='i2h at ' .. l}
        local h2h = nn.Linear(hidden_dim, 4 * hidden_dim)(prev_h):annotate{name='h2h at ' .. l}
        local preacts = nn.CAddTable()({i2h, h2h}):annotate{name='pre-activation at ' .. l}
        local direction = 2
        local all_gates = nn.Sigmoid()(nn.Narrow(direction, 1, 3*hidden_dim)(preacts))
        all_gates:annotate{name='all gates at ' .. l}
        local update = nn.Tanh()(nn.Narrow(direction, 3*hidden_dim + 1, hidden_dim)(preacts))
        update:annotate{name='trans at' .. l}
        local i_gate = nn.Narrow(direction, 1, hidden_dim)(all_gates)
        i_gate:annotate{name='input gates at ' .. l}
        local f_gate = nn.Narrow(direction, hidden_dim + 1, hidden_dim)(all_gates)
        f_gate:annotate{name='forget gates at ' .. l}
        local o_gate = nn.Narrow(direction, 2 * hidden_dim + 1, hidden_dim)(all_gates)
        o_gate:annotate{name='output gates at ' .. l}
        c[l] = nn.CAddTable()({
                nn.CMulTable()({f_gate, prev_c}),
                nn.CMulTable()({i_gate, update})
            }):annotate{name='c at' .. l}
        h[l] = nn.CMulTable()({
                o_gate,
                nn.Tanh()(c[l])
            }):annotate{name='h at' .. l}
    end
    local extra_output = output_unit(h[num_layers])
    -- local decoder = nn.Linear(hidden_dim, num_classes)(h[num_layers]):annotate{name='decoder'}
    -- local logdist = nn.LogSoftMax()(decoder)
    local outputs = {nn.Identity()(c), nn.Identity()(h), extra_output} 
    local cell = nn.gModule(inputs, outputs)
    return cell
end

function LSTM.lite_lstm(config)
    local input_dim = config.input_dim
    local hidden_dim = config.hidden_dim
    local num_layers = config.num_layers
    local input_X = nn.Identity()():annotate{name='input'}
    local prev_C = nn.Identity()():annotate{name='prev C'}
    local prev_H = nn.Identity()():annotate{name='prev H'}
    local inputs = {input_X, prev_C, prev_H}
    local c, h, outputs = {}, {}, {}
    for l = 1, num_layers do
        local prev_c = nn.SelectTable(l)(prev_C):annotate{name='prev c at ' .. l}
        local prev_h = nn.SelectTable(l)(prev_H):annotate{name='prev h at ' .. l}
        local dim, input
        if l == 1 then
            input = input_X
            dim = input_dim
        else
            input = h[l-1]
            dim = hidden_dim
        end
        local i2h = nn.Linear(dim, 4 * hidden_dim)(input):annotate{name='i2h at ' .. l}
        local h2h = nn.Linear(hidden_dim, 4 * hidden_dim)(prev_h):annotate{name='h2h at ' .. l}
        local preacts = nn.CAddTable()({i2h, h2h}):annotate{name='pre-activation at ' .. l}
        local direction = 2
        local all_gates = nn.Sigmoid()(nn.Narrow(direction, 1, 3*hidden_dim)(preacts))
        all_gates:annotate{name='all gates at ' .. l}
        local update = nn.Tanh()(nn.Narrow(direction, 3*hidden_dim + 1, hidden_dim)(preacts))
        update:annotate{name='trans at' .. l}
        local i_gate = nn.Narrow(direction, 1, hidden_dim)(all_gates)
        i_gate:annotate{name='input gates at ' .. l}
        local f_gate = nn.Narrow(direction, hidden_dim + 1, hidden_dim)(all_gates)
        f_gate:annotate{name='forget gates at ' .. l}
        local o_gate = nn.Narrow(direction, 2 * hidden_dim + 1, hidden_dim)(all_gates)
        o_gate:annotate{name='output gates at ' .. l}
        c[l] = nn.CAddTable()({
                nn.CMulTable()({f_gate, prev_c}),
                nn.CMulTable()({i_gate, update})
            }):annotate{name='c at' .. l}
        h[l] = nn.CMulTable()({
                o_gate,
                nn.Tanh()(c[l])
            }):annotate{name='h at' .. l}
    end
    local outputs = {nn.Identity()(c), nn.Identity()(h)} 
    local cell = nn.gModule(inputs, outputs)
    return cell
end

function LSTM.share_params(sharing, target)
    if torch.isTypeOf(target, 'nn.gModule') then
        for i = 1, #target.forwardnodes do
            local node = target.forwardnodes[i]
            if node.data.module then
                LSTM.share_params(
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

function LSTM.duplicate_cells(master_cell, num)
    local cells = {}
    local mem = torch.MemoryFile("w"):binary()
    mem:writeObject(master_cell)
    for i = 1, num do
        local reader = torch.MemoryFile(mem:storage(), "r"):binary()
        cells[i] = reader:readObject()
        reader:close()
        LSTM.share_params(master_cell, cells[i])
    end
    mem:close()
    collectgarbage()
    return cells
end

function LSTM.init_states(batch_size, num_layers, hidden_dim, cuda)
    local init_C, init_H = {}, {}
    local values = torch.zeros(batch_size, hidden_dim)
    if cuda then
        values = values:cuda()
    end
    for l = 1, num_layers do
        init_C[l] = values
        init_H[l] = values
    end
    return {init_C, init_H}, values
end

return LSTM