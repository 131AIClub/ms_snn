from mindspore.nn import Adam, MSELoss
from mindspore.train import Model, LossMonitor
from mindspore import context
from mindspore.dataset import GeneratorDataset
# from mindspore import value_and_grad, save_checkpoint
# import mindspore.ops as ops

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

# def forward_fn(inputs):
#     inputs = inputs.transpose((1, 0, 2, 3, 4))
#     logits = []
#     for t in range(inputs.shape[0]):
#         x = inputs[t]
#         y = net(x)
#         logits.append(y.unsqueeze(0))
#     logits = ops.mean(ops.cat(logits, 0), 0)

#     return logits


# grad_fn = value_and_grad(lambda input, target: loss_fn(
#     forward_fn(input), target), None, optimizer.parameters)

# for epoch in range(num_epochs):
#     losses = []
#     net.set_train()
#     epoch_start = time.time()

# # 为每轮训练读入数据
#     net.set_train(True)
#     for i, (images, labels) in enumerate(data_loader_train):
#         loss, grads = grad_fn(images, labels)
#         optimizer(grads)
#         losses.append(loss)
#         if i % 10 == 0:
#             print(f"train step {i}, loss: {loss}")

#     # 每个epoch结束后，验证准确率
#     net.set_train(False)
#     correct_num = 0
#     total_num = 0
#     for images, labels in data_loader_test:
#         logits = forward_fn(images)
#         pred = logits.argmax(axis=1)
#         labels = labels.argmax(axis=1)
#         correct = ops.equal(pred, labels)
#         correct_num += correct.sum().asnumpy()
#         total_num += correct.shape[0]
#     acc = correct_num / total_num

#     epoch_end = time.time()
#     epoch_seconds = (epoch_end - epoch_start) * 1000
#     step_seconds = epoch_seconds/step_size_train

#     print("-" * 20)
#     print("Epoch: [%3d/%3d], Average Train Loss: [%5.3f], Accuracy: [%5.3f]" % (
#         epoch+1, num_epochs, sum(losses)/len(losses), acc
#     ))
#     print("epoch time: %5.3f ms, per step time: %5.3f ms" % (
#         epoch_seconds, step_seconds
#     ))

#     if acc > best_acc:
#         best_acc = acc
#         if not os.path.exists(best_ckpt_dir):
#             os.mkdir(best_ckpt_dir)
#         save_checkpoint(net, best_ckpt_path)

# print("=" * 80)
# print(f"End of validation the best Accuracy is: {best_acc: 5.3f}, "
#       f"save the best ckpt file in {best_ckpt_path}", flush=True)
