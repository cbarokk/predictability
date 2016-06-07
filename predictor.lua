
require 'torch'
require 'nn'
require 'nngraph'
require 'sys'

require 'util.misc'
require 'SequenceFactory'

local NextEvent = require 'NextEvent'


cmd = torch.CmdLine()
cmd:text()
cmd:text('Predict the next event in a stream. Waits for file paths to be pushed on <redis_prefix>-predictions')
cmd:text()
cmd:text('Options')
cmd:option('-len_seq',50,'number of timesteps to unroll for')
cmd:option('-batch_size',100,'number of sequences to train on in parallel')
cmd:option('-redis_prefix', '', 'redis key name prefix, where to read train/validation/events data')
cmd:option('-window_size',60,'width of window for weighted averaging of time predictions')
cmd:option('-data', 'data', 'path to data when sequence source is disk')

cmd:option('-init_from', '', 'initialize network parameters from checkpoint at this path')
cmd:option('-sequence_provider', 4,'which sequence provider to use, 1 for redis, 2 for synthetic, 3 from disc, 4 for disk_sequential')

-- GPU/CPU
cmd:option('-gpuid',-1,'which gpu to use. -1 = use CPU')
cmd:option('-verbose',1,'set to 0 to ONLY print the sampled text, no diagnostics')

cmd:text()

-- parse input params
opt = cmd:parse(arg)

-- create the data loader class  

opt.disc_access = "sequential"
opt.sequence_provider = 3

init()

local loader = NextEvent.create()

lstm_init()

-- initialize the rnn state to all zeros

current_state = init_state()

print ("starting predictor")
protos.rnn:evaluate() -- put in eval mode so that BatchNormalization and DropOut works properly
batch_type = "testing" -- only look at test data

local function get_score(prediction, truth)
  local score = prediction:gather(2, truth:view(opt.batch_size,1))
  score = score:expand(prediction:size()[1], prediction:size()[2])
  local top_score = torch.gt(prediction, score):sum(2)
  return top_score:view(opt.batch_size):float()
end

local function get_iqr(seq_score, q1, q3)
  local score = torch.sort(seq_score:float(), 1)
  local iqr_idx = torch.LongTensor({q1,q3}):view(2,1):expand(2, score:size()[2]) 
  local q1_q3 = score:gather(1, iqr_idx)
  return q1_q3:mean(1)
end

local count = 0

while true do
  collectgarbage()
  count = count + 1
  -- fetch a batch
  local x, y, e_x, e_y = loader:next_batch()
  local len_seq = x:size()[1]
  
  if len_seq > 5 then
    local q1 = math.ceil(len_seq*0.25)
    local q3 = math.floor(len_seq*0.75)
  
    local seq_x_score = torch.DoubleTensor(len_seq, opt.batch_size)
    local seq_ex_score = torch.DoubleTensor(len_seq, opt.batch_size)
  
    if opt.gpuid > -1 then 
      seq_x_score = seq_x_score:cuda()
      seq_ex_score = seq_ex_score:cuda()
    end
  
    local current_state = init_state()
    -- forward pass
    local lst
    for t=1, len_seq do
      lst = protos.rnn:forward{x[t], e_x[t], unpack(current_state)}
      seq_x_score[t] = get_score(lst[#lst-1], y[t])

      seq_ex_score[t] = get_score(lst[#lst], e_y[t])
      current_state = {}
      for i=1,2*#opt.rnn_layers do table.insert(current_state, lst[i]) end
    end
    
    local iqr_x = get_iqr(seq_x_score, q1, q3):div(opt.num_time_slots)[1]
    local iqr_ex = get_iqr(seq_ex_score, q1, q3):div(opt.num_time_slots)[1]
  
    -- todo: redis_client:zadd("predictability:" .. loader.sources[1], avg_scores[j], seq)

  
    local avg_scores = 0.5*(iqr_x + iqr_ex)
    --print (avg_scores)
  
    for j=1,#loader.batch do
      print (count, loader.sources[j], len_seq)

      local seq = ""  
      for i=1,#loader.batch[j]-1 do
        seq = seq .. "," .. loader.batch[j][i][1] .. "-" .. opt.event_inv_mapping[tonumber(loader.batch[j][i][2])]
      end
      --print (avg_scores[j], seq)
      redis_client:zadd("predictability", avg_scores[j], seq)
    end
  end
  
end



--[[
    if t > 1 then
      print ("e_x[t]", opt.event_inv_mapping[e_x[t][1] ], opt.event_inv_mapping[last_pred])
    else 
      print ("e_x[t]", opt.event_inv_mapping[e_x[t][1] ])
    end
    
    lst = protos.rnn:forward{x[t], e_x[t], unpack(current_state)}
   
    local e_y_probs = lst[#lst]
    local e_y_probs_sorted, e_y_probs_idx = torch.sort(e_y_probs, 2, true) -- sort along 2nd dim, true = in descending order
    
    local y_probs = lst[#lst-1]
    --smooth_probs(y_probs, opt.window_size)
    
    -- The predictions for the events are in e_y_probs_sorted (the indices are in e_y_probs_idx) 
    -- The predictions for the time slots are in y_probs (not sorted)
    
    
    
    last_pred = e_y_probs_idx[1][1]

]]--


