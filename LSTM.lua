-- This LSTM is based on Adam Paszke's tutorial http://apaszke.github.io/lstm-explained.html
-- Encapsulation of forward, backward, and cell creation are based on StanfordNLP's TreeLSTM
-- https://github.com/stanfordnlp/treelstm/blob/master/models/LSTM.lua
-- Author: NorThanapon
-- Date: Oct 07, 2015

-- Input: A tensor containing sequence of inputs (input_dim X T)
-- Output: A table containing all hidden states in the sequence (hidden_dim X num_layers X T)
local LSTM, parent = torch.class('lstm.LSTM', 'nn.Module')

-- Construct LSTM with
---- num_layers: #hidden layers
---- input_dim: #dimensions of the input
---- hidden_dim: #dimensions of the hidden state and cell state
-- TODO:
---- Optimize for forward only mode (don't keep rolled-out cells)
function LSTM:__init(config)
    parent.__init(self)
    -- setting hyper-params
    self.input_dim = config.input_dim
    self.hidden_dim = config.hidden_dim
    self.num_layers = config.num_layers
    
    -- create master cell (all cells will share master cell's params)
    self.master_cell = self:new_cell()
    -- depth to keep track of current state
    self.depth = 0
    -- roll out cells
    self.cells = {} 
    
    -- Setting initial values
    local c_init, h_init, c_grad, h_grad
    if self.num_layers == 1 then
        c_init = torch.zeros(self.hidden_dim)
        h_init = torch.zeros(self.hidden_dim)
        c_grad = {torch.zeros(self.hidden_dim)}
        h_grad = {torch.zeros(self.hidden_dim)}
    else
        c_init, h_init, c_grad, h_grad = {}, {}, {}, {}
        for l = 1,self.num_layers do
            c_init[l] = torch.zeros(self.hidden_dim)
            h_init[l] = torch.zeros(self.hidden_dim)
            c_grad[l] = torch.zeros(self.hidden_dim)
            h_grad[l] = torch.zeros(self.hidden_dim)
        end
    end
    self.init_values = {c_init, h_init}
    -- grad_input for backward prob
    self.grad_input = {
        torch.zeros(self.input_dim),
        c_grad,
        h_grad
    }
end
    
function LSTM:new_cell()
    local input = nn.Identity()()
    local c_p = nn.Identity()()
    local h_p = nn.Identity()()
    local inputs = {input, c_p, h_p} -- {x_t, c_{t-1}, h_{t-1}}
    local h, c = {}, {}
    
    for l = 1, self.num_layers do
        local c_l_p = nn.SelectTable(l)(c_p) -- c_{t-1}{l}
        local h_l_p = nn.SelectTable(l)(h_p) -- h_{t-1}{l}
        local i2h
        if l == 1 then
            i2h = nn.Linear(self.input_dim, 4*self.hidden_dim)(input) -- W_x * x_t + b_x
        else
            i2h = nn.Linear(self.hidden_dim, 4*self.hidden_dim)(h[l-1]) -- W_x * h_{t}{l-1} + b_x
        end
        local h2h = nn.Linear(self.hidden_dim, 4*self.hidden_dim)(h_l_p) -- W_h * h_{t-1}{l} + b_h  
        -- preactivations for i_t, f_t, o_t, c_in_t (update)
        local preacts = nn.CAddTable()({i2h, h2h}) -- i2h + h2h
    
        -- direction of Narrow = 1 (column vector input)
        -- nonlinear:
        --     input, forget, and output gates get Sigmoid
        --     state update gets Tanh
        local all_gates = nn.Sigmoid()(nn.Narrow(1, 1, 3*self.hidden_dim)(preacts)) 
        local update = nn.Tanh()(nn.Narrow(1, 3*self.hidden_dim + 1, self.hidden_dim)(preacts))
        -- split gates into their variables
        local i_gate = nn.Narrow(1, 1, self.hidden_dim)(all_gates)
        local f_gate = nn.Narrow(1, self.hidden_dim + 1, self.hidden_dim)(all_gates)
        local o_gate = nn.Narrow(1, 2 * self.hidden_dim + 1, self.hidden_dim)(all_gates)
        -- new state, c_{t}{l} = f_t .* c_{t-1}{l} + i_t .* c_in_t
        c[l] = nn.CAddTable()({
                nn.CMulTable()({f_gate, c_l_p}),
                nn.CMulTable()({i_gate, update})
            })
        -- new hidden, h_{t}{l} = o_t .* Tanh(c)
        h[l] = nn.CMulTable()({
                o_gate,
                nn.Tanh()(c[l])
            })
    end
    -- output new state c, and new hidden h
    -- c and h are a table. the output will have the size of 2 x num_layers
    -- c = outputs[1], h = outputs[2] 
    local outputs = {nn.Identity()(c), nn.Identity()(h)} 
    local cell = nn.gModule(inputs, outputs)
    -- It is important that all cells share the same set of params (same pointer)
    if self.master_cell then
        lstm.utils.share_parameters(self.master_cell, cell)
    end
    return cell
end

-- inputs: input_dim x T
-- outputs: hidden_dim X num_layers X T
function LSTM:forward(inputs)
    local size = inputs:size(2)
    local out_states = torch.Tensor(self.hidden_dim, self.num_layers, size)
    for t = 1, size do
        -- set up input and depth
        local input = inputs[{{},t}]
        self.depth = self.depth + 1
        -- reuse/create cell for this time step
        local cell = self.cells[self.depth]
        if cell == nil then
            cell = self:new_cell()
            self.cells[self.depth] = cell
        end
        -- get previous states
        local prev_output
        if self.depth > 1 then
            prev_output = self.cells[self.depth - 1].output
        else
            prev_output = self.init_values
        end
        -- in case of 1 layer, the cell will output 2 tensors of c and h
        -- instead of 2 tables of tensors, so we need to wrap the table for c and h
        prev_c = prev_output[1]
        prev_h = prev_output[2]
        if self.num_layers == 1 then
            prev_c = {prev_c}
            prev_h = {prev_h}
        end
        -- forward prop
        local outputs = cell:forward({input, prev_c, prev_h})        
        -- output
        self.output = outputs
        if self.num_layers == 1 then
            out_states[{{},{},t}] = outputs[2]
        else
            for i=1,self.num_layers do
                out_states[{{}, i, t}] = outputs[2][i]
            end
        end
    end
    return out_states
end

-- inputs = input_dim x T
-- grad_output = hiddin_dim x num_layers x T
function LSTM:backward(inputs, grad_outputs)
    local size = inputs:size(2)
    local input_grads = torch.Tensor(inputs:size(1), inputs:size(2))
    -- backward from last forwarded cell to the first one
    for t = size,1,-1 do
        -- get input and cell for current step
        local input = inputs[{{},t}]
        local cell = self.cells[self.depth]
        -- get gradient of the output (2nd argurment)
        local grad_output = grad_outputs[{{},{},t}]        
        -- recover previous states (t-1)
        local prev_output
        if self.depth > 1 then
            prev_output = self.cells[self.depth - 1].output
        else
            prev_output = self.init_values
        end
        local prev_c = prev_output[1]
        local prev_h = prev_output[2]
        -- get gradients of the t+1 cell (previous iteration)
        local grad_c = self.grad_input[2]
        local grad_h = self.grad_input[3]
        if self.num_layers == 1 then
            prev_c = {prev_c}
            prev_h = {prev_h}
        end
        -- add output gradient to the hidden state gradients
        for i = 1, self.num_layers do
            grad_h[i]:add(grad_output[{{},i}])
        end
        local grads
        if self.num_layers == 1 then
            grads = {grad_c[1], grad_h[1]}
        else
            grads = {grad_c, grad_h}
        end
        -- compute gradients of the input and keep it for the next iteration
        self.grad_input = cell:backward({input, prev_c, prev_h}, grads)
        -- keep the gradient for returning
        input_grads[{{},t}] = self.grad_input[1]
        -- go back one step
        self.depth = self.depth - 1
    end
    -- reset
    self:forget()
    return input_grads
end

function LSTM:zeroGradParameters()
  self.master_cell:zeroGradParameters()
end

function LSTM:parameters()
  return self.master_cell:parameters()
end

-- reset depth of the network and all gradients to zero
function LSTM:forget()
  self.depth = 0
  for i = 1, #self.grad_input do
    local grad_input = self.grad_input[i]
    if type(grad_input) == 'table' then
      for _, t in pairs(grad_input) do t:zero() end
    else
      self.grad_input[i]:zero()
    end
  end
end