-- This LSTM is based on Adam Paszke's tutorial http://apaszke.github.io/lstm-explained.html
-- Encapsulation of forward, backward, and cell creation are based on StanfordNLP's TreeLSTM
-- Author: NorThanapon
-- Date: Oct 07, 2015

-- Input: A tensor containing sequence of inputs (input_dim X T)
-- Output: A table containing all cell and hidden states in the sequence {hidden_dim, hidden_dim} X T
local LSTM, parent = torch.class('lstm.LSTM', 'nn.Module')

-- Construct LSTM with
---- input_dim: #dimensions of the input
---- hidden_dim: #dimensions of the hidden state and cell state
-- TODO:
---- Allow each cell can be multi-layer network
---- Optimize for forward only mode (don't keep rolled-out cells)
function LSTM:__init(config)
    parent.__init(self)
    -- setting hyper-params
    self.input_dim = config.input_dim
    self.hidden_dim = config.hidden_dim
    
    -- create master cell (all cells will share master cell's params)
    self.master_cell = self:new_cell()
    -- depth to keep track of current state
    self.depth = 0
    -- roll out cells
    self.cells = {} 
    
    -- Setting initial values
    local c_init, h_init, c_grad, h_grad
    c_init = torch.zeros(self.hidden_dim)
    h_init = torch.zeros(self.hidden_dim)
    c_grad = torch.zeros(self.hidden_dim)
    h_grad = torch.zeros(self.hidden_dim)
    self.init_values = {c_init, h_init}
    -- gradInput for backward prob
    self.gradInput = {
        torch.zeros(self.input_dim),
        c_grad,
        h_grad
    }
end
    
function LSTM:new_cell()
    local inputs = {} -- {x_t, c_{t-1}, h_{t-1}}
    table.insert(inputs, nn.Identity()())
    table.insert(inputs, nn.Identity()())
    table.insert(inputs, nn.Identity()())
    local input = inputs[1]
    local c_p = inputs[2]
    local h_p = inputs[3]
    local i2h = nn.Linear(self.input_dim, 4*self.hidden_dim)(input) -- W_x * x_t + b_x
    local h2h = nn.Linear(self.hidden_dim, 4*self.hidden_dim)(h_p) -- W_h * h_{t-1} + b_h
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
    -- new state, c = f_t .* c_p + i_t .* c_in_t
    local c = nn.CAddTable()({
            nn.CMulTable()({f_gate, c_p}),
            nn.CMulTable()({i_gate, update})
        })
    -- new hidden, h = o_t .* Tanh(c)
    local h = nn.CMulTable()({
            o_gate,
            nn.Tanh()(c)
        })
    -- output new state c, and new hidden h
    local outputs = {}
    table.insert(outputs, c)
    table.insert(outputs, h)
    local cell = nn.gModule(inputs, outputs)
    
    -- It is important that all cells share the same set of params (same pointer)
    if self.master_cell then
        lstm.utils.share_parameters(self.master_cell, cell)
    end
    return cell
end

-- inputs: input_dim x T
-- outputs: {hidden_dim, hidden_dim} X T (cell state, and hidden state)
function LSTM:forward(inputs)
    local size = inputs:size(2)
    local out_stats = {}
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
        -- forward prop
        local outputs = cell:forward({input, prev_output[1], prev_output[2]})        
        -- output
        self.output = outputs
        out_stats[t] = outputs
    end
    return out_stats
end

-- inputs = input_dim x T
-- grad_output = hiddin_dim x T
function LSTM:backward(inputs, grad_outputs)
    local size = inputs:size(2)
    local input_grads = torch.Tensor(inputs:size(1), inputs:size(2))
    -- backward from last forwarded cell to the first one
    for t = size,1,-1 do
        -- get input and cell for current step
        local input = inputs[{{},t}]
        local cell = self.cells[self.depth]
        -- get gradient of the output (2nd argurment)
        local grad_output = grad_outputs[{{},t}]
        -- get gradients of the t+1 cell (one time step forward)
        local grads = {self.gradInput[2], self.gradInput[3]}
        -- add output gradient to the hidden state gradients
        grads[2]:add(grad_output)
        -- recover previous states (t-1)
        local prev_output
        if self.depth > 1 then
            prev_output = self.cells[self.depth - 1].output
        else
            prev_output = self.init_values
        end
        -- compute gradients of the input and keep it for the next iteration (used:132)
        self.gradInput = cell:backward({input, prev_output[1], prev_output[2]}, grads)
        -- keep the gradient for returning
        input_grads[{{},t}] = self.gradInput[1]
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
  for i = 1, #self.gradInput do
    local gradInput = self.gradInput[i]
    if type(gradInput) == 'table' then
      for _, t in pairs(gradInput) do t:zero() end
    else
      self.gradInput[i]:zero()
    end
  end
end