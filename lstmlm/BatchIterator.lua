local BatchIterator = torch.class('BatchIterator')

function BatchIterator:__init(data, batch_size, circular)
    self.data = data
    self.batch_size = batch_size
    self.circular = circular
    self.t = 1
    self.done = false
end

function BatchIterator:has_next()
    return not self.done
end

function BatchIterator:reset()
    self.done = false
    self.t = 1
end

function BatchIterator:next_batch()
    local idx
    self.done, self.t, idx = self._next_batch(self.t, self.data, self.batch_size, self.circular)
    return idx
end


function BatchIterator._next_batch(t, shuffle_index, batch_size, circle)
    local lbatch_size = math.min(t + batch_size - 1, shuffle_index:size(1)) - t + 1
    local idx = shuffle_index[{{t, t + lbatch_size - 1}}]
    local done = false
    t = t + idx:size(1)
    if lbatch_size < batch_size and circle then
        local full_idx = torch.zeros(batch_size)
        full_idx[{{1,lbatch_size}}] = idx
        t = 1
        _, t, idx = BatchIterator._next_batch(t, shuffle_index, batch_size - lbatch_size)
        full_idx[{{lbatch_size + 1, batch_size}}] = idx
        return true, t, full_idx
    end
    if t > shuffle_index:size(1) then
        done = true
    end
    return done, t, idx
end

function BatchIterator.shuffle_seq(batch_size, num_batches)
    local shuffle_start_idx = torch.randperm(num_batches)
    local shuffle_idx = torch.zeros(data:size(1))
    local ib = 1
    for i = 1, num_batches do
        start_idx = (i - 1) * batch_size + 1
        end_idx = start_idx + batch_size - 1
        local shift = shuffle_start_idx[ib]
        for j = start_idx, end_idx do
            shuffle_idx[j] = shift
            shift = shift + num_batches
        end
        ib = ib + 1
        if ib > num_batches then ib = (ib % num_batches) + 1 end
    end
    return shuffle_idx
end