import os
import re
import shutil
import time
from datetime import datetime, timedelta
import torch
from torch.nn import DataParallel
from tqdm import tqdm
import numpy as np
import logging

from src.utils import getDevice
from src.utils import getData
from src.model import MusicRWKV, MusicRWKVConfig


def newTrain(device_ids, device):
    # 初始化epoch数
    last_epoch = 0
    # 构建模型
    model = MusicRWKV(MusicRWKVConfig(257, 257, model_type="RWKV",
                                      n_layer=6, n_embd=256)).to(device)
    # 如果有多个GPU，那么就多个GPU一起训练
    if len(device_ids) > 1:
        model = DataParallel(model, device_ids=device_ids, output_device=device_ids[0])
    # 获取优化方法
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return model, optimizer, last_epoch


def resumeTrain(device_ids, device):
    pretrained_model_path = os.path.join(save_model_dir, pretrained_dir)
    # 获取预训练的epoch数
    last_epoch = int(re.findall('(\d+)', pretrained_model_path)[-2])
    model = MusicRWKV(MusicRWKVConfig(257, 257, model_type="RWKV",
                                      n_layer=6, n_embd=256)).to(device)
    if len(device_ids) > 1:
        pass
    else:
        model.load_state_dict(torch.load(os.path.join(pretrained_model_path, 'model.pth')))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer.load_state_dict(torch.load(os.path.join(pretrained_model_path, 'optimizer.pth')))
    logger.info(f"成功加载模型参数和优化方法参数 {pretrained_dir}")
    return model, optimizer, last_epoch


# 测试模型
@torch.no_grad()
def modelTest(model, device, epoch_id, batch_id, test_loader):
    model.eval()
    logger.info('=' * 100)
    accuracies = []
    for spec_mel, label in test_loader:
        spec_mel = spec_mel.to(device)
        label = label.to(device)
        pred = model(spec_mel)
        pred = pred.data.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        label = label.data.cpu().numpy()
        acc = np.mean((pred == label).astype(int))
        accuracies.append(acc.item())
    accuracy = float(sum(accuracies) / len(accuracies)) if len(accuracies) != 0 else float(0)
    model.train()
    logger.info(f"Test epoch {epoch_id} batch {batch_id} Accuracy {accuracy:.5}")
    logger.info('=' * 100)


# 保存模型
def modelSave(model, optimizer, epoch_id, batch_id):
    model_params_path = os.path.join(save_model_dir, f"epoch_{epoch_id}_batch_{batch_id}")
    if not os.path.exists(model_params_path):
        os.makedirs(model_params_path)
    # 保存模型参数和优化方法参数配置
    torch.save(model.state_dict(), os.path.join(model_params_path, 'model.pth'))
    torch.save(optimizer.state_dict(), os.path.join(model_params_path, 'optimizer.pth'))
    # 删除旧的模型
    old_model_path = os.path.join(save_model_dir, f"epoch_{(epoch_id - 5)}_batch_{batch_id}")
    if os.path.exists(old_model_path):
        shutil.rmtree(old_model_path)

########################################################################################################
# # Train model
########################################################################################################

np.set_printoptions(precision=4, suppress=True, linewidth=200)
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, )

gpus = "0"
batch_size = 32
num_workers = 8
num_epoch = 100
learning_rate = 1e-4
train_list_path = "data/train_list.txt"
test_list_path = "data/test_list.txt"
test_output_dir = "test_output/"
save_model_dir = "trained_models/"
resume = True
pretrained_dir = "epoch_12_batch_30/"

logger = logging.getLogger(__name__)
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
os.environ['RWKV_FLOAT_MODE'] = 'bf16'
device_ids = [int(i) for i in gpus.split(',')]
device = getDevice()
# 获取数据集
train_loader, test_loader = getData(train_list_path, test_list_path, batch_size, device_ids, num_workers)
# 加载模型参数和优化方法参数
# 这里是加载一下预训练模型
if resume:
    model, optimizer, last_epoch = resumeTrain(device_ids, device)
else:
    model, optimizer, last_epoch = newTrain(device_ids, device)

# 获取损失函数
criterion = torch.nn.CrossEntropyLoss()

# 开始训练
sum_batch = len(train_loader) * (num_epoch - last_epoch)
# proc_bar = tqdm(total=sum_batch)
start = time.time()
model.train()
for epoch_id in range(last_epoch, num_epoch):
    epoch_id += 1
    batch_id = 0
    # 获取我们的输入和标签
    for spec_mel, label in train_loader:
        batch_id += 1
        spec_mel = spec_mel.to(device)
        label = label.to(device)
        # 先调用restNet获取特征值
        pred = model(spec_mel)
        # 计算loss
        loss = criterion(pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 每迭代XX次就打印一下准确率和loss信息
        if batch_id % 10 == 0:
            # eta_sec = (time.time() - start) * 1000 * (
            #         sum_batch - (epoch_id - last_epoch) * len(train_loader) - batch_id)
            spd_sec=str(timedelta(seconds=int(time.time()) - int(start)))
            logger.info(f"Train epoch {epoch_id}, batch: {batch_id}, loss: {loss.item()}, lr: {learning_rate}, sum_batch: {sum_batch}, spd_time: {spd_sec}")
            # proc_bar.set_description(
            #     f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Train epoch {epoch_id}, batch: {batch_id}, loss: {loss.item()}, lr: {learning_rate}, sum_batch: {sum_batch}, spd_time: {spd_sec}")
            # proc_bar.update(10)
        # 每迭代XX次就保存模型
        if batch_id % 30 == 0:
            # 开始测试
            modelTest(model, device, epoch_id, batch_id, test_loader)
            # 保存模型
            if len(device_ids) > 1:
                modelSave(model.module, optimizer, epoch_id, batch_id)
            else:
                modelSave(model, optimizer, epoch_id, batch_id)

            # proc_bar.close()
