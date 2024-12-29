from net import CIFAR10DVSNet
import numpy as np
import mindspore as ms
from mindspore import context
from mindspore.train import Model

net = CIFAR10DVSNet()
context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")


def load_npz_frames(file_name: str) -> np.ndarray:
    '''
        :param file_name: path of the npz file that saves the frames
        :type file_name: str
        :return: frames
        :rtype: np.ndarray
        '''
    return np.load(file_name, allow_pickle=True)['frames'].astype(np.float32)


sample = load_npz_frames(
    "/home/dataset/DVSCifar10/airplane/cifar10_airplane_1.npz")
loss_fn = ms.nn.MSELoss()
optimizer = ms.nn.Adam(net.trainable_params(), learning_rate=0.1)

model = Model(net, loss_fn=loss_fn,
              optimizer=optimizer, metrics={'accuracy'})

output = net(ms.tensor(sample))
loss = loss_fn(output, ms.tensor([1.0, 0.0]))
optimizer = ms.nn.Adam(net.trainable_params())


def forward_fn(inputs, targets):

    logits = net(inputs)
    print(logits)
    loss = loss_fn(logits, targets)

    return loss


grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters)


def train_step(inputs, targets):

    loss, grads = grad_fn(inputs, targets)
    print("loss: "+str(loss))
    print("grad: "+str(grads))
    optimizer(grads)

    return loss, grads


loss, grad = train_step(ms.tensor(sample), ms.tensor([1.0, 0.0,]))
for p in net.trainable_params():
    print(p)
