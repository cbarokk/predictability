# predictability
Code to train and test predictability models

**training**

th learner.lua -redis_prefix topups -sequence_provider 3 -seed `date +%s`

**predictions**

th predictor.lua -init_from model_gro.t7 -redis_prefix topups
