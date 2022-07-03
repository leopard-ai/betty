Memory Optimization
===================

As MLO involves multiple optimization problems, it tends to use more GPU memory than
traditional single-level optimization problems. Given the recent trend of large models
(e.g., Transformers, BERT, GPT), Betty strives to support large-scale MLO and
meta-learning by implementing various memory optimization techniques.

Currently, we support three memory optimization techniques:

1. Gradient accumulation.
2. FP16 (half-precision) training.
3. (Non-distributed) Data-parallel training.

|

Setup
-----

To better analyze the effects of memory optimization, we scale up our dataset and
classifier network (from our data reweighting example in :doc:`basic_start`) to CIFAR10
and ResNet50. As this is not directly relevant to this tutorial, users can copy and
paste code from the collapsable code snippets below.

.. raw:: html

   <details>
   <summary><a>Long-tailed CIFAR10 Dataset</a></summary>

.. code-block:: python

    from torchvision.datasets import CIFAR10


    device = "cuda" if torch.cuda.is_available() else "cpu"

    def build_dataset(reweight_size=1000, imbalanced_factor=100):
        normalize = transforms.Normalize(
            mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
            std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
        )
        transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )

        dataset = CIFAR10(root="./data", train=True, download=True, transform=transform)

        num_classes = len(dataset.classes)
        num_meta = int(reweight_size / num_classes)

        index_to_meta = []
        index_to_train = []

        imbalanced_num_list = []
        sample_num = int((len(dataset.targets) - reweight_size) / num_classes)
        for class_index in range(num_classes):
            imbalanced_num = sample_num / (imbalanced_factor ** (class_index / (num_classes - 1)))
            imbalanced_num_list.append(int(imbalanced_num))
        np.random.shuffle(imbalanced_num_list)

        for class_index in range(num_classes):
            index_to_class = [
                index for index, label in enumerate(dataset.targets) if label == class_index
            ]
            np.random.shuffle(index_to_class)
            index_to_meta.extend(index_to_class[:num_meta])
            index_to_class_for_train = index_to_class[num_meta:]

            index_to_class_for_train = index_to_class_for_train[: imbalanced_num_list[class_index]]

            index_to_train.extend(index_to_class_for_train)

        reweight_dataset = copy.deepcopy(dataset)
        dataset.data = dataset.data[index_to_train]
        dataset.targets = list(np.array(dataset.targets)[index_to_train])
        reweight_dataset.data = reweight_dataset.data[index_to_meta]
        reweight_dataset.targets = list(np.array(reweight_dataset.targets)[index_to_meta])

        return dataset, reweight_dataset


    classifier_dataset, reweight_dataset = build_dataset(reweight_size=1000, imbalanced_factor=100)
    normalize = transforms.Normalize(
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
    )
    valid_transform = transforms.Compose([transforms.ToTensor(), normalize])
    valid_dataset = CIFAR10(root="./data", train=False, transform=valid_transform)
    valid_dataloader = DataLoader(valid_dataset, batch_size=100, pin_memory=True)

.. raw:: html

   </details>

.. raw:: html

   <details>
   <summary><a>ResNet50 Classifier</a></summary>

.. code-block:: python

    class Bottleneck(nn.Module):
        expansion = 4

        def __init__(self, in_planes, planes, stride=1):
            super(Bottleneck, self).__init__()
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(self.expansion * planes)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != self.expansion * planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False
                    ),
                    nn.BatchNorm2d(self.expansion * planes),
                )

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = F.relu(self.bn2(self.conv2(out)))
            out = self.bn3(self.conv3(out))
            out += self.shortcut(x)
            out = F.relu(out)
            return out


    class ResNet(nn.Module):
        def __init__(self, block, num_blocks, num_classes=10):
            super(ResNet, self).__init__()
            self.in_planes = 64

            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
            self.linear = nn.Linear(512 * block.expansion, num_classes)

        def _make_layer(self, block, planes, num_blocks, stride):
            strides = [stride] + [1] * (num_blocks - 1)
            layers = []
            for stride in strides:
                layers.append(block(self.in_planes, planes, stride))
                self.in_planes = planes * block.expansion
            return nn.Sequential(*layers)

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            return out


    def ResNet50():
        return ResNet(Bottleneck, [3, 4, 6, 3])


    classifier_module = ResNet50()

.. raw:: html

   </details>

|

Gradient Accumulation
---------------------

Gradient accumulation is an effective way to reduce the memory of intermediate states by
accumulating gradients from *multiple small* mini-batches rather than calculating the
gradient of *one large* mini-batch. In Betty, gradient accumulation can be enabled for
*each level* problem via ``Config`` as:

.. code:: python

    reweight_config = Config(type="darts", log_step=100, gradient_accumulation=4)

|

FP16 Training
-------------

FP16 (or half-precision) training replaces some of the full-precision operations (e.g.
linear) with half-precision operations at the cost of (potential) training instability.
In Betty, users can easily enable FP16 training for *each level* problem via ``Config``
as:

.. code:: python

    reweight_config = Config(type="darts", log_step=100, gradient_accumulation=4, fp16=True)

|

(Non-distributed) Data-parallel training
----------------------------------------

Data-parallel training splits large batches into several small batches across multiple
GPUs and thereby reduces the memory footprint for intermediate states.  While
distributed data-parallel training normally achieves much better training speed, Betty
currently only supports non-distributed data-parallel training via ``EngineConfig``:

.. code:: python

    engine_config = EngineConfig(train_iters=10000, valid_step=100, distributed=True)

|

Memory optimization results
---------------------------
We perform an ablation study to analyze how each technique affects GPU memroy usage.
The result is shown in the table below.

+--------------+--------------+
|              | Memory       |
+==============+==============+
| Baseline     | 6817MiB      |
+--------------+--------------+
| +FP16        | 4397MiB      |
+--------------+--------------+
| +Distributed | 3185/3077MiB |
+--------------+--------------+

For the distributed setting, we report two memory usages (one for each GPU).
