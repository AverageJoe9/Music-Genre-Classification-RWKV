from torch.utils import data
from torch.utils.data import DataLoader
import torch
import numpy as np
import os
import librosa
import librosa.feature
import json

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def getDevice():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using ' + str(device))
    return device


# 加载并预处理音频
def loadAudio(audio_path, sr=16000, spec_len=257):
    # 读取音频数据
    wav, sr = librosa.load(path=audio_path, sr=sr)
    # 计算短时傅里叶变换
    linear = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=1024, hop_length=512, n_mels=256)
    melspec = librosa.power_to_db(linear, ref=np.max)
    freq, freq_time = melspec.shape
    assert freq_time >= spec_len
    # 随机裁剪
    rand_time = np.random.randint(0, freq_time - spec_len)
    spec_melspec = melspec[:, rand_time:rand_time + spec_len]
    # 归一化
    mean = np.mean(spec_melspec, 0, keepdims=True)
    std = np.std(spec_melspec, 0, keepdims=True)
    spec_mag = (spec_melspec - mean) / (std + 1e-5)
    spec_mag=spec_mag.swapaxes(0,1)
    return spec_mag


def getData(train_list_path, test_list_path, batch_size, device_ids, num_workers):
    train_loader = DataLoader(dataset=CustomDataset(train_list_path),
                              batch_size=batch_size * len(device_ids),
                              shuffle=True,
                              num_workers=num_workers)
    test_loader = DataLoader(dataset=CustomDataset(test_list_path), batch_size=batch_size, num_workers=num_workers)
    return train_loader, test_loader


class CustomDataset(data.Dataset):
    def __init__(self, data_list_path, spec_len=257):
        super(CustomDataset, self).__init__()
        with open(data_list_path, 'r') as f:
            datalist = f.read()
        self.datalist = json.loads(datalist)
        self.spec_len = spec_len

    def __getitem__(self, idx):
        audio_path, idx = self.datalist[idx]
        spec_mel = loadAudio(audio_path, spec_len=self.spec_len)
        label = np.array(int(idx), dtype=np.int64)
        # 这里会返回一个特征信息和用户的标签
        return spec_mel, label

    def __len__(self):
        return len(self.datalist)


if __name__ == '__main__':
    # CustomDataset("data/train_list.txt")
    loadAudio("../data/train/jazz/jazz.00000.wav")

