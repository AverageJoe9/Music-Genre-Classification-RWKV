# Music-Genre-Classification-RWKV
___
>A music genre classification model for ten genres: blues, classical, country, disco, hip-hop, jazz, metal, pop, reggae, rock.

References:

https://github.com/xiaoyou-bilibili/voice_recognize

https://github.com/yeyupiaoling/VoiceprintRecognition-Pytorch

https://github.com/BlinkDL/RWKV-LM

Please install dependencies first.
```bash
pip install requirements.txt
```
Please install PyTorch corresponding to your own CUDA and Python version.

Please create directories named 'data', 'models' and 'inference'.
## About Training
Dataset: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/data

After downloading, separate the various styles into 'train' and 'test' parts, and place them respectively in 'data/train' and 'data/test' folders.

Directory Structure
```
├── data
  ├── train
    ├──blues
      ├──audio1
      ├──audio2
    ├──classical
      ├──audio1
      ├──audio2
    ...
  ├── test
    ├──blues
      ├──audio1
      ├──audio2
    ├──classical
      ├──audio1
      ├──audio2
    ...
```
Then set the parameters in the 'train' section of the config.yaml file.

Run
```bash
python preprocess.py
python train.py
```
## About Inference
Please set the 'audio_path' in the 'inference' section of config.yaml to the path of the audio file for inference.

Then run
```bash
python inference.py
```
## About Model
Net: RWKV-v4

Loss function: CrossEntropyLoss

Optimizer：Adam