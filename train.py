from mindspore.nn import Adam, MSELoss
from mindspore.train import Model, LossMonitor
from mindspore import context
from mindspore.dataset import GeneratorDataset

from net import CIFAR10DVSNet
from dataset import DVSSource

context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

# 最佳模型保存路径
best_ckpt_dir = "./BestCheckpoint"
best_ckpt_path = "./BestCheckpoint/resnet50-best.ckpt"
# 数据集路径
dataset_path = "/home/dataset/DVSCifar10_copy"
class_num = 4

# 定义训练轮数
best_acc = 0
num_epochs = 100

# 开始循环训练
print("Start Training Loop ...")

# 定义模型
net = CIFAR10DVSNet(class_num=class_num)

# 加载数据集
set = GeneratorDataset(source=DVSSource(dataset_path, class_num=class_num), column_names=[
                       'data', 'label'], shuffle=False)
train_ds, test_ds = set.split([0.9, 0.1])
# shuffle训练集并batch
train_ds = train_ds.shuffle(train_ds.get_dataset_size())
train_ds, test_ds = train_ds.batch(4), test_ds.batch(4)

data_loader_train = train_ds.create_tuple_iterator(num_epochs=num_epochs)
data_loader_test = test_ds.create_tuple_iterator(num_epochs=num_epochs)
step_size_train = train_ds.get_dataset_size()

# 定义损失函数、优化器
loss_fn = MSELoss()
optimizer = Adam(net.trainable_params(), learning_rate=0.0005)
model = Model(net, loss_fn=loss_fn,
              optimizer=optimizer, metrics={'accuracy'})
loss_moniter = LossMonitor(10)

model.fit(10, train_ds, test_ds, 1, loss_moniter)