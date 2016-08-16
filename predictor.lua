
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

local function get_event_score(preds, index_truth)
  local s = -math.log(0.005)
  local score = preds:gather(2, index_truth:view(opt.batch_size,1)):mul(-1):div(s)
  return score
end

local function get_AUC(y)
    local area = torch.DoubleTensor(opt.batch_size):fill(0)
    local N = y:size()[2]
    for i = 2, N do
      area:add(y:sub(1,-1, i-1,i):clone():sum(2):div(2))  --simplified formula because all intervals are of width 1
    end
    area:div(N-1)  --divide by the number of intervals to scale AUC to 0-1  
    return area
end 

local function get_time_score(preds, index_truth)
  -- Look at probabilities to the left and right of the truth 
 
  preds:exp()
  
  local tmp = torch.cat({preds, preds, preds},2)
  local N = opt.num_time_slots
  -- Find cumulative probabilities at increasing distance from the truth
  
  for i=1, opt.batch_size do
    preds:sub(i,i, 1, N/2):copy(tmp:sub(i,i, N/2+index_truth[i], N+index_truth[i]-1))
    preds:sub(i,i, N/2+1, -1):copy(tmp:sub(i,i, N+index_truth[i], N+index_truth[i]+N/2-1))
  end
  
  local cum_probs = torch.DoubleTensor(opt.batch_size, N/2+1)
    
  cum_probs:sub(1,-1, 1,1):copy(preds:sub(1,-1, N/2+1, N/2+1))
  
  for i=2, N/2 do
    cum_probs:sub(1,-1, i,i):copy(cum_probs:sub(1,-1, i-1,i-1)):add(preds:sub(1,-1, N/2+1 - (i-1), N/2+1 - (i-1))):add(preds:sub(1,-1, N/2+1 + i-1, N/2+1 + i-1))
  end  
  cum_probs:sub(1,-1, -1, -1):fill(1)
  
  -- Calculate AUC scaled to 0-1 interval
  local auc = get_AUC(cum_probs) -- I used a built-in function in R using the 'trapezoidal rule'; I think this can be implemented in lua as below
  local score = 1-auc
  
  return score


end



local count = 0

while true do
  collectgarbage()
  count = count + 1
  -- fetch a batch
  local x, y, e_x, e_y = loader:next_batch()
  local len_seq = x:size()[1]
  
  local time_score_f = torch.DiskFile('scores/time_score_batch_' .. count, 'w')
  local event_score_f = torch.DiskFile('scores/event_score_batch_' .. count, 'w')
  
  print ("batch", count)
  
  if len_seq > 5 then
    local event_pred_score = torch.DoubleTensor(len_seq, opt.batch_size)
    local time_pred_score = torch.DoubleTensor(len_seq, opt.batch_size)
  
    if opt.gpuid > -1 then 
      event_pred_score = event_pred_score:cuda()
      time_pred_score = time_pred_score:cuda()
    end
  
    local current_state = init_state()
    -- forward pass
    local lst
    for t=1, len_seq do
      lst = protos.rnn:forward{x[t], e_x[t], unpack(current_state)}
      
      -- Calculate predictability scores for each event in each sequence --
      event_pred_score[t] = get_event_score(lst[#lst], e_y[t]) 
      time_pred_score[t] = get_time_score(lst[#lst-1], y[t])
  
      current_state = {}
      for i=1,2*#opt.rnn_layers do table.insert(current_state, lst[i]) end
    end
  
    for j=1,#loader.batch do
      local id = string.gsub(loader.sources[j]:split("/")[2], "%s+", "")
      
      time_score_f:writeString(id)
      event_score_f:writeString(id)
      if j < #loader.batch then
        time_score_f:writeString(", ")
        event_score_f:writeString(", ")
      end
    end
    time_score_f:writeString("\n")
    event_score_f:writeString("\n")
  
    for t=1, opt.len_seq-1 do
      for j=1,#loader.batch do
        if j>1 then
          time_score_f:writeString(", ")
          event_score_f:writeString(", ")
        end
        time_score_f:writeString(tostring(time_pred_score[t][j]))
        event_score_f:writeString(tostring(event_pred_score[t][j]))
      end
      time_score_f:writeString("\n")
      event_score_f:writeString("\n")
    end
  end
end
