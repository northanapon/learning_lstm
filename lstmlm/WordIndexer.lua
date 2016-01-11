--[[

Bi-directional indexer for word <-> index

]]--

local WordIndexer = torch.class('WordIndexer')

function WordIndexer:__init()
    self.w2idx = {}
    self.idx2w = {}
    self.size = 0
    self.unk_idx = -1
end

function WordIndexer:__len()
    return self.size
end

function WordIndexer:contain(w)
    if self.w2idx[w] then
        return true
    end
    return false
end

function WordIndexer:set(idx, w)
    if not self.w2idx[w] then
        self.size = self.size + 1
    end
    self.w2idx[w] = idx
    self.idx2w[idx] = w
end

function WordIndexer:index(w)
    local idx = self.w2idx[w]
    if not idx then
        return self.unk_idx
    end
    return idx
end

function WordIndexer:indexes(words)
    local indexes = torch.IntTensor(#words)
    for i = 1, #words do
        indexes[i] = self:index(words[i])
    end
    return indexes
end

function WordIndexer:word(i)
    return self.idx2w[i]
end

function WordIndexer:words(indexes)
    local words = {}
    for i = 1, #indexes do
        words[i] = self:word(indexes[i])
    end
    return words
end

function WordIndexer:add(w)
    if self.w2idx[w] then
        return self.w2idx[w]
    end
    self.size = self.size + 1
    self.w2idx[w] = self.size
    self.idx2w[self.size] = w
    return self.size
end