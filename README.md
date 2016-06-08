# predictability
Code to train and test predictability models

## requirements
The code requires [Torch](http://torch.ch/)
See the Torch installation documentation for more details. After Torch is installed we need to get a few more packages using [LuaRocks](https://luarocks.org/) (which already came with the Torch install). In particular:

```bash
$ luarocks install nngraph
$ luarocks install optim
$ luarocks install nn
$ luarocks install redis-lua
```

Then, you need to patch 2 files in the nn package for BatchNormalization to work with our cloned rnn implementation.

```bash
$ cp nn_path/Container.lua /<path>/<to>/torch/install/share/lua/5.1/nn/Container.lua
$ cp nn_path/BatchNormalization.lua /<path>/<to>/torch/install/share/lua/5.1/nn/BatchNormalization.lua
```

## training

th learner.lua -redis_prefix topups -sequence_provider 3 -seed `date +%s`

## predictions

th predictor.lua -init_from cv/model_lstm_512_1_1_epoch510.00_7.32616564.t7_cpu.t7 -redis_prefix topups
