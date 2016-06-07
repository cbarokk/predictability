# predictability
Code to train and test predictability models

## requirements
The code requires [Torch](http://torch.ch/)
See the Torch installation documentation for more details. After Torch is installed we need to get a few more packages using [LuaRocks](https://luarocks.org/) (which already came with the Torch install). In particular:

```bash
$ luarocks install nngraph
$ luarocks install optim
$ luarocks install nn
```



## training

th learner.lua -redis_prefix topups -sequence_provider 3 -seed `date +%s`

## predictions

th predictor.lua -init_from cv/model_lstm_512_1_1_epoch510.00_7.32616564.t7_cpu.t7 -redis_prefix topups
