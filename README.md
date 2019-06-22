# Question-NLI

This is the course project of F033569 NLU of SJTU.

# Requirements

* Python3.*
* Pytorch 0.4.1 (not test in higher version)
* [pytorch_pretrained_bert](https://github.com/huggingface/pytorch-pretrained-BERT)
* tqdm (to monitor the training progress)
* GPU (indispensable, since the model is saved with CUDA format)

# To train

Put your data into the /data folder and the pretrained model in /data/.cache folder, then simply run

```
sh train.sh
```

or

```
python3 main.py --mode train --lower_case --gpu --gpu_id 0 --max_seq_len 128 --num_epochs 20 --lr 2e-5 --batch_size 32 --warmup_proportion 0.1
```

# To test

Set the paths of your models in `model_list.txt` (note that the file should be end with a new line), then run

```
sh test.sh
```