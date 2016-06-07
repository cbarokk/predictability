--require 'util.MyDropout'

local LSTM_NextEvent = {}

function LSTM_NextEvent.lstm()
  local num_time_slots = opt.num_time_slots
  local num_events = opt.num_events
  local dropout = opt.dropout or 0
  local lambda = opt.lambda
  local rnn_layers = opt.rnn_layers
  
  -- there will be 2*n+2 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  table.insert(inputs, nn.Identity()()) -- e_x
  
  local event_embed_table

    function LSTM_Module(input_size, input, prev_c, prev_h, output_size, annotation)
      -- evaluate the input sums at once for efficiency
      local i2h = nn.Linear(input_size, 4 * output_size)(input):annotate{name='i2h_'..annotation}
      local h2h = nn.Linear(output_size, 4 * output_size)(prev_h):annotate{name='h2h_'..annotation}
      local all_input_sums = nn.CAddTable()({i2h, h2h})
      all_input_sums = nn.BatchNormalization(4*output_size)(all_input_sums)
      
      local reshaped = nn.Reshape(4, output_size)(all_input_sums)
      local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
      -- decode the gates
      local in_gate = nn.Sigmoid()(n1):annotate{name='in_gate_'..annotation}
      local forget_gate = nn.Sigmoid()(n2):annotate{name='forget_gate_'..annotation}
      local out_gate = nn.Sigmoid()(n3):annotate{name='out_gate_'..annotation}
      -- decode the write inputs
      local in_transform = nn.Tanh()(n4)
      -- perform the LSTM update
      local next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
      }):annotate{name='next_c_'..annotation}
      -- gated cells form the output
      
      local next_c_h = nn.BatchNormalization(output_size)(next_c)
      
      local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c_h)}):annotate{name='next_h_'..annotation}
      return next_c, next_h
    end


  for L = 1,#rnn_layers do
    table.insert(inputs, nn.Identity()()) -- prev_c[L]
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  local input_x, input_ex, input_size_x, input_size_ex
  local outputs = {}
  local embedings_size = 100
  
  local next_h_x, next_h_ex 
  local next_c_x, next_c_ex 
    
  for L = 1,#rnn_layers do

  -- c,h from previos timesteps
    local prev_c = inputs[L*2+1]
    local prev_h = inputs[L*2+2]
    
    prev_h = nn.Reshape(2, rnn_layers[L]/2)(prev_h)
    local prev_h_x, prev_h_ex = nn.SplitTable(2)(prev_h):split(2)
    prev_c = nn.Reshape(2, rnn_layers[L]/2)(prev_c)
    local prev_c_x, prev_c_ex = nn.SplitTable(2)(prev_c):split(2)

    -- the input to this layer
    if L == 1 then 
      input_x = inputs[1]
      input_ex = inputs[2]
      
      input_ex = nn.LookupTable(num_events, embedings_size)(input_ex):annotate{name='emb_e'}
      input_ex = nn.Reshape(embedings_size)(input_ex)
      
      event_embed_table = input_ex
      
      input_size_x = theta_size
      input_size_ex = embedings_size
      
    else 
      input_x = next_h_x
      input_ex = next_h_ex
      
      --if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      input_size_x = rnn_layers[L-1]/2
      input_size_ex = rnn_layers[L-1]/2
  
    end

  --if L == 2 then
      if dropout > 0 then input_ex = nn.Dropout(dropout)(input_ex) end -- apply dropout, if any
      if dropout > 0 then input_x = nn.Dropout(dropout)(input_x) end -- apply dropout, if any
  --end


    next_c_x, next_h_x = LSTM_Module(input_size_x, input_x, prev_c_x, prev_h_x, rnn_layers[L]/2, L)
    next_c_ex, next_h_ex = LSTM_Module(input_size_ex, input_ex, prev_c_ex, prev_h_ex, rnn_layers[L]/2, L)
    
    local next_c = nn.JoinTable(2)({next_c_x, next_c_ex}) 
    local next_h = nn.JoinTable(2)({next_h_x, next_h_ex}) 


    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end
  
  local layer = outputs[#outputs]

  local layer_size = rnn_layers[#rnn_layers]
  
  
  local time_pred = nn.Linear(layer_size, num_time_slots)(layer):annotate{name='softmax time'}
  time_pred = nn.LogSoftMax()(time_pred)
  table.insert(outputs, time_pred)
  
  local proj = nn.Linear(layer_size, num_events)(layer):annotate{name='softmax events'}
  local logsoft = nn.LogSoftMax()(proj)
  table.insert(outputs, logsoft)
  
  return nn.gModule(inputs, outputs)
end

return LSTM_NextEvent

