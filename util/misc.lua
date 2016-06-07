
-- misc utilities
require 'gnuplot'

local redis = require 'redis'
redis_client = redis.connect('127.0.0.1', 6379)


function clone_list(tensor_list, zero_too)
    -- utility function. todo: move away to some utils file?
    -- takes a list of tensors and returns a list of cloned tensors
    local out = {}
    for k,v in pairs(tensor_list) do
        out[k] = v:clone()
        if zero_too then out[k]:zero() end
    end
    return out
end

function get_size_softmax_layer(forwardnodes, annotation)
  for _,node in ipairs(forwardnodes) do
     if node.data.annotations.name == annotation then
      return node.data.module.weight:size(1)
    end
  end
end

function Welch(N)
  local w = torch.Tensor(N)
  local i = -1
  local half = (N-1)/2
  w:apply(function()
    i = i + 1   
    return 1-(math.pow((i-half)/half,2))
  end)
  w:div(w:sum())
  return w
end

local welch_table = {}

function smooth_probs(probs, N)
  if welch_table[N] == nil then
    local welch_batch = torch.DoubleTensor(probs:size()[1], N*2+1)
    local w = Welch(2*N+1)
    
    if opt.gpuid >= 0 then -- ship the input arrays to GPU
    -- have to convert to float because integers can't be cuda()'d
      w = w:float():cuda()
      welch_batch = welch_batch:float():cuda()
    end
    for i=1, probs:size()[1] do
      welch_batch:sub(i,i):copy(w)
    end
    welch_table[N] = welch_batch
  end
  
  local w = welch_table[N]
  --local left_half_w = w:sub(1,N/2+1):div(w:sub(1,N/2+1):sum())
  
  local tmp = torch.cat({
      probs:sub(1,-1, probs:size()[2] - N+1, -1), 
      probs,
      probs:sub(1,-1, 1, N)},2)
  if opt.gpuid >= 0 then -- ship the input arrays to GPU
    tmp = tmp:float():cuda()
  end
  
  local offset = N
    for i = 1, probs:size()[2] do
      probs:sub(1,-1,i,i):fill(torch.cmul(tmp:sub(1,-1,offset+i-N, offset+i+N),w):sum())
    end
  probs:div(probs:sum()) -- normalize so it sums to 1
end
  
function normal_equations(X, y)
  return torch.inverse(X:t()*X)*X:t()*y
end


theta_size = 8

function timestamp2theta(timestamp)
  local theta = torch.DoubleTensor(theta_size):fill(0)
  
  local date = os.date("*t", timestamp)
  
  
  local sec = date['sec']
  theta[1] = math.cos((2*math.pi)/60*sec) --cos_sec
  theta[2] = math.sin((2*math.pi)/60*sec) --sin_sec
  
  local min = date['min']
  theta[3] = math.cos((2*math.pi)/60*min) --cos_min
  theta[4] = math.sin((2*math.pi)/60*min) --sin_min
      
  local hour = date['hour']
  theta[5] = math.cos((2*math.pi)/24*hour) --cos_hour
  theta[6] = math.sin((2*math.pi)/24*hour) --sin_hour
      
  local weekday = date['wday']-1
  theta[7] = math.cos((2*math.pi)/7*weekday) --cos_weekday
  theta[8] = math.sin((2*math.pi)/7*weekday) --sin_weekday
  --[[
  local monthday = date['day']
  theta[9] = math.cos((2*math.pi)/31*monthday) --cos_monthday
  theta[10] = math.sin((2*math.pi)/31*monthday) --sin_monthday

  local month = date['month']
  theta[11] = math.cos((2*math.pi)/12*month) --cos_month
  theta[12] = math.sin((2*math.pi)/12*month) --sin_month
  
  local yearday = date['yday']
  theta[13] = math.cos((2*math.pi)/365*yearday) --cos_yearday
  theta[14] = math.sin((2*math.pi)/365*yearday) --sin_yearday
]]--
  return theta, date
end

function PCA(X)
  local mean = torch.mean(X, 1) -- 1 x n
  local m = X:size(1)
  local Xm = X - torch.ones(m, 1) * mean
  
  Xm:div(math.sqrt(m - 1))
  local v,s,_ = torch.svd(Xm:t())
  
  s:cmul(s) -- n

  --[[
  -- v: eigenvectors, s: eigenvalues of covariance matrix
  local b = sys.COLORS.blue; n = sys.COLORS.none
  print(b .. 'eigenvectors (columns):' .. n); print(v)
  print(b .. 'eigenvalues (power/variance):' .. n); print(s)
  print(b .. 'sqrt of the above (energy/std):' .. n); print(torch.sqrt(s))
  ]]--
  
  local vv = v * torch.diag(torch.sqrt(s))
  vv = torch.cat(torch.ones(2,1) * mean, vv:t())
  return vv
  
end

function display_uncertainty(y_pred, y_fasit, e_y_pred, e_y_fasit)
      local y_probs = torch.exp(y_pred[1]):float()
      local y_truth = y_fasit:clone()[1]
      
      local s=0
      --local left_cumul = y_probs:clone():mean(1)
      local left_cumul = y_probs:clone()
      
      left_cumul:apply(function(x)
          s = s + x
          return s
        end)
      
      --local right_cumul = y_probs:clone():mean(1)
      local right_cumul = y_probs:clone()
      s=0
      for i=y_probs:size()[1], 1, -1 do
        s = s + right_cumul[i]
        right_cumul[i] = s
      end
      local y_truth_dist = left_cumul:clone():zero()
      
      y_truth_dist[y_truth] =  y_probs:max()
        
      --y_truth_dist:div(y_truth_dist:sum())
      
      
      gnuplot.figure('horizon')
      gnuplot.title('horizon ')
      gnuplot.axis{1, y_probs:size()[1],0, y_probs:max()}
      --gnuplot.plot({"cpdf (left)", left_cumul,'-'}, {"y", y_truth_dist, '|'}, {"bets", y_probs, '-'}, {"cpdf (right)", right_cumul,'-'})
      gnuplot.plot({"y", y_truth_dist, '|'}, {"bets", y_probs, '-'})
      gnuplot.raw('set xtics ("Sun" 1, "06:00" 360, "12:00" 720, "18:00" 1080, "Mon" 1440, "06:00" 1800, "12:00" 2160, "18:00" 2520, "Tue" 2880, "06:00" 3240, "12:00" 3600, "18:00" 3960,"Wed" 4320, "06:00" 4680, "12:00" 5040, "18:00" 5400, "Thu" 5760, "06:00" 6120, "12:00" 6480, "18:00" 6840, "Fri" 7200, "06:00" 7560, "12:00" 7920, "18:00" 8280, "Sat" 8640, "06:00" 9000, "12:00" 9360, "18:00" 9720)')

      --sys.sleep(3)
--[[
      local e_y_probs = torch.exp(e_y_pred):float()
      local e_y_truth = e_y_probs:clone():zero()[1]
      
      e_y_truth[e_y_fasit[1] ] = e_y_probs:max()
      
      gnuplot.axis{1, e_y_probs:size()[1],0,e_y_probs:max()}
      
      gnuplot.figure('next event')
      gnuplot.title('next event ')
      gnuplot.plot({"e_y", e_y_truth, '|'}, {"bets", e_y_probs[1], '-'})
  ]]--    
      
    end
    
    
  function fetch_events()
    if string.len(opt.init_from) > 0 then --
      print "recovering options from checkpoint"
      opt.num_events = checkpoint.opt.num_events
      opt.event_mapping = checkpoint.opt.event_mapping
      opt.event_inv_mapping = checkpoint.opt.event_inv_mapping
      opt.time_slot_size = checkpoint.opt.time_slot_size
      opt.horizon = checkpoint.opt.horizon
      opt.num_time_slots = checkpoint.opt.num_time_slots
    else
      opt.num_events = redis_client:scard(opt.redis_prefix .. '-events') + 1
      local tmp = redis_client:smembers(opt.redis_prefix .. '-events')
      opt.num_events = # tmp + 1
      opt.event_mapping = {}
      opt.event_inv_mapping = {}

      for id, event_id in pairs(tmp) do
        opt.event_mapping[event_id] = id+1 -- 1 reserved to "not intersting"
        opt.event_inv_mapping[id+1] = event_id
      end
      opt.event_inv_mapping[1] = "PROBE"
      opt.time_slot_size = opt.horizon/opt.num_time_slots
    end

end

function string.starts(String,Start)
   return string.sub(String,1,string.len(Start))==Start
end

function dump_tensor(annotation, key, file)
  for _,node in ipairs(protos.rnn.forwardnodes) do
    if node.data.annotations.name == annotation then
      local x= node.data.module.output:float()
      file:writeString(key ..": ")
      file:writeFloat(x:storage())
    end
  end
end
    
    
  
  
function init_state()
  local state={}
  for L=1, #opt.rnn_layers do
    local h_init = torch.zeros(opt.batch_size, opt.rnn_layers[L])
    if opt.gpuid >=0 then h_init = h_init:cuda() end
    table.insert(state, h_init:clone())
    if opt.rnn_unit == 'lstm' then
        table.insert(state, h_init:clone())
    end
  end
  return state
end

local function gpu_init()
   -- initialize cunn/cutorch for training on the GPU and fall back to CPU gracefully
  if opt.gpuid >= 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then print('package cunn not found!') end
    if not ok2 then print('package cutorch not found!') end
    if not (ok and ok2) then
      print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
      print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
      print('Falling back on CPU mode')
      opt.gpuid = -1 -- overwrite user setting
    end
  end
end


function recover_opt_from_checkpoint()
  
  opt.do_random_init = true
  if string.len(opt.init_from) > 0 then
    print('loading an LSTM from checkpoint ' .. opt.init_from)
    checkpoint = torch.load(opt.init_from)
    protos = checkpoint.protos

    -- overwrite model settings based on checkpoint to ensure compatibility
    print('overwriting options from checkpoint.')
    print ("last learning rate", checkpoint.learning_rate)
    opt.rnn_layers = checkpoint.opt.rnn_layers
    opt.rnn_unit = checkpoint.opt.rnn_unit
    opt.seed = checkpoint.opt.seed
    opt.start_epoch = checkpoint.epoch
    opt.do_random_init = false
  else
    opt.rnn_layers = {unpack(opt.layer_sizes:split(","))}
    for i =1, #opt.rnn_layers do
      opt.rnn_layers[i] = tonumber(opt.rnn_layers[i])
    end
    opt.start_epoch = 0
  end
  
  torch.manualSeed(opt.seed)

  if opt.gpuid >= 0 then
    print('using CUDA on GPU ' .. opt.gpuid .. '...')
    cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
    cutorch.manualSeed(opt.seed)
  end
end


function lstm_init()
  -- initialize the LSTM forget gates with slightly higher biases to encourage remembering in the beginning
  if opt.rnn_unit == 'lstm' then
    for layer_idx = 1, #opt.rnn_layers do
      for _,node in ipairs(protos.rnn.forwardnodes) do
        if node.data.annotations.name == "i2h_" .. layer_idx then
          print('setting forget gate biases to 1 in LSTM layer ' .. layer_idx)
          -- the gates are, in order, i,f,o,g, so f is the 2nd block of weights
          node.data.module.bias[{{opt.rnn_layers[layer_idx]+1, 2*opt.rnn_layers[layer_idx]}}]:fill(1.0)
        end
      end
    end
  end
  
  -- ship the model to the GPU if desired
  if opt.gpuid >= 0 then
    for k,v in pairs(protos) do v:cuda() end
  end
end

function init()
  gpu_init()
  recover_opt_from_checkpoint()
  factory_init()
end
  