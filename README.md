## Train msimcse-embed-method
```
python train.py
```
We train with num_epochs = 50 and lr=1e-4.
Dataset: I use the dataset wanhin/msimcse_512_seqlen on hugging face and will be loaded in the train file.
Model msimcse: wanhin/msimcse_vi-en on hugging face and will be loaded in the train file. We don't train this model.
Model mt5: google/mt5-base on hugging face and will be loaded in the train file. We train this model.

# Test model
```
python test.py
```

Model msimcse: wanhin/msimcse_vi-en
Model mt5: #new_checkpoint!!!#
