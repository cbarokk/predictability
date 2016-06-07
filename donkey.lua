require 'lfs'
local BUFSIZE = 2^14     -- 16K

local SEQ_DELIM = "./"

local training_data = {}
local validation_data = {}
local testing_data = {}

local data_file

local function loadChunk(file, data, bufsize, seek)
  if seek then file:seek("set", seek) end
  local buf_size = buf_size or BUFSIZE
  local buf, rest = file:read(buf_size, "*line")
  if not buf then return false end
  if rest then buf = buf .. rest .. '\n' end
  table.insert(data, buf)
  return true
end

local function parseLine(chunk, lines)
  for line in chunk:gmatch("[^\r\n]+") do 
    table.insert(lines, line)
  end
end
  

local function parseLines(data)
  local lines={}
  for _, chunk in pairs(data) do
    parseLine(chunk, lines)
  end
  return lines
end


local function loadSeq(sample)
  local seek = sample[2]
  local bufsize = sample[3]
  data_file:seek("set", seek)
  local buf, rest = data_file:read(bufsize, "*line")
  local seq = {}
  parseLine(buf, seq)
  --local lines = {}
  --parseLine(buf, lines)
  --print ("lines", lines)
  --local seq = {}
  --for i, line in pairs(lines) do
  --  if string.starts(line, SEQ_DELIM) then return seq  end
  --  table.insert(seq, line)
  --end
  return seq
end



local function loadFile(path) 
  local f = io.input(path)
  local data = {}
  while loadChunk(f, data) do end
  return parseLines(data)
end  

local function splitLines(lineTable)
  for i=1, #lineTable do
    lineTable[i] = string.split(lineTable[i], ",")
    lineTable[i][2] = opt.event_mapping[lineTable[i][2]]
  end
end


function loadData()
  data_file = io.input(opt.data .. '/data')
  print ("Loading data indexes ...")
  lines = loadFile(opt.data .. "/data_idx.csv", true)
  for i=1, #lines do
    lines[i] =  string.split(lines[i], ",")
    lines[i][2] = tonumber(lines[i][2])
    lines[i][3] = tonumber(lines[i][3])-lines[i][2]
  end
  splitData(lines)

end

function splitData(lines)
  local total = #lines
  if opt.disc_access == "sequential" then 
    testing_data = lines
  end
  
  if opt.disc_access == "random" then   
    print ("Splitting into traing/validation/testing sets...")
    while #lines > 0 do
      line = table.remove(lines)
      local r = torch.uniform()
        
      if r <= 0.6 then table.insert(training_data, line) end
      if 0.6 < r and r <= 0.8 then table.insert(validation_data, line) end
      if 0.8 < r then table.insert(testing_data, line) end
    end
  end
  
  local output = string.format('Distribution -- training:%.2f, validation:%.2f, testing:%.2f',
    #training_data/total, #validation_data/total, #testing_data/total)
  print (output)
    
end


function loadEvents()
  local path = opt.data .. '/events'

  local lines = loadFile(path, true)
  for i=1, #lines do
    redis_client:sadd(opt.redis_prefix .. "-events", lines[i])
  end
end


local function splitLines2(lineTable, keep_k)
  local pos = 1
  
  if #lineTable > keep_k then
    pos = torch.random(1, #lineTable - keep_k)
  end
  
  local tmp ={}
  
  for i=1, math.min(keep_k, #lineTable) do
    table.insert(tmp, string.split(lineTable[pos+i-1], ","))
    tmp[i][2] = opt.event_mapping[tmp[i][2]]
  end
  return tmp
end

local function insertProbes(seq, len_seq)
  while #seq < len_seq do
    local pos = torch.random(1, #seq-1)
    local t = torch.random(seq[pos][1], seq[pos+1][1])
    table.insert(seq, pos+1, {tostring(t), 1})
  end
  
end 

function loadBatch(size, len_seq)
  local batch = {}
  local sources = {}

  while #batch < size do 
    local sample_table
    
    if batch_type == "training" then
      sample_table = training_data
    end
    if batch_type == "validation" then
      sample_table = validation_data
    end
    if batch_type == "testing" then
      sample_table = testing_data
    end
    
    
    local pos = torch.random(1, #sample_table)
    
    local sample = sample_table[pos]
    local seq = loadSeq(sample)
    
    if #seq > math.ceil(len_seq*1.2) then
        seq = splitLines2(seq, len_seq)
        --insertProbes(seq, len_seq)
      table.insert(sources, sample[1])
      table.insert(batch, seq)
    else
      table.remove(sample_table, pos)
    end
  end
  
  --print ("sources", sources)
  --print ("batch", batch)
  
  
  collectgarbage()
  return batch, sources
end


function nextBatch(size, len_seq)
  local batch = {}
  local sources = {}

  while #batch < size and #testing_data > 0 do 
    local sample = table.remove(testing_data)
    local seq = loadSeq(sample)
    if #seq > math.ceil(len_seq) then
      seq = splitLines2(seq, len_seq)
      table.insert(sources, sample[1])
      table.insert(batch, seq)
    end
  end
  collectgarbage()
  return batch, sources
end

