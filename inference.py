import os

import torch
import numpy as np

from src.utils import getDevice
from src.utils import loadAudio
from src.model import MusicRWKV,MusicRWKVConfig

save_model_dir = "trained_models/"
pretrained_dir = "epoch_7_batch_30/"
audio_path = './inference/落日飞车 - 我是一只鱼.mp3'
genre_list = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

os.environ['RWKV_FLOAT_MODE'] = 'bf16'
device = getDevice()
# 加载模型
model = MusicRWKV(MusicRWKVConfig(257, 257, model_type="RWKV",
                                      n_layer=6, n_embd=256)).to(device)
pretrained_model_path = os.path.join(save_model_dir, pretrained_dir)
model.load_state_dict(torch.load(os.path.join(pretrained_model_path, 'model.pth')))
model.to(device)
model.eval()
print("Using model: " + pretrained_dir)
# 加载音频
spec_mel = loadAudio(audio_path)
spec_mel = spec_mel[np.newaxis, :]
spec_mel = torch.tensor(spec_mel, dtype=torch.float32, device=device)
print("Audio path: " + audio_path)
# 执行预测
pred = model(spec_mel)
pred = pred.data.cpu().numpy()[0]
pred = np.argmax(pred)
print("Inference result: " + genre_list[pred])
