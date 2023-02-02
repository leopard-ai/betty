import torch
import torch.nn as nn
from torch.utils.model_zoo import load_url as load_state_dict_from_url
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

# from soft_argmax import SoftArgmax1D

__all__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "wide_resnet50_2",
    "wide_resnet101_2",
    "RNNDecoder",
    "Learner",
]


model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.block = block

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x_in = torch.flatten(x, 1)
        # x = self.fc(x_in)
        # x_v = self.fc_v(x_in)

        return x_in

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    # print(model)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        # print(state_dict)
        state_dict.pop("fc.weight", None)
        state_dict.pop("fc.bias", None)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
            progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
            progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet34", BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
            progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
            progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        "resnet101", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs
    )


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
            progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        "resnet152", Bottleneck, [3, 8, 36, 3], pretrained, progress, **kwargs
    )


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
            progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    return _resnet(
        "resnext50_32x4d", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs
    )


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
            progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 8
    return _resnet(
        "resnext101_32x8d", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs
    )


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
            progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet(
        "wide_resnet50_2", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs
    )


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
            pretrained (bool): If True, returns a model pre-trained on ImageNet
            progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet(
        "wide_resnet101_2", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs
    )


#  you can activate it or not using self.requires_grad.


class HotEmbedding(torch.nn.Module):
    def __init__(self, max_val, embedding_dim, eps=1e-4):
        super(HotEmbedding, self).__init__()
        self.A = torch.arange(max_val, requires_grad=False).cuda()
        self.B = torch.randn((max_val, embedding_dim), requires_grad=True).cuda()
        self.eps = eps
        self.max_val = max_val
        self.embedding_dim = embedding_dim

    def forward(self, x):
        # print(x.shape, self.A.shape, self.B.shape)
        # print(self.A, self.B)
        ans = 1 / ((x.unsqueeze(-1) ** 2 - self.A**2) + self.eps) @ self.B
        # print(ans)
        return ans

    # def to_one_hot(self, inp):
    # 	ones = torch.eye(self.max_val)
    # 	vecs =torch.stack([torch.select_index(ones,0,batch) for batch in inp.detach().int()],0)
    # 	vecs.requires_grad = True
    # 	for batch in range(inp.size(0)):
    # 		for i in range(inp.size(1)):
    # 			vecs[batch][inp[batch][i]] =


def diff_argmax(vec):
    ans = vec.argmax(1).detach() + vec.max(1)[0] - vec.max(1)[0].detach()
    # print("argmax: ",ans)
    return ans


class celoss:
    def __init__(self, vocab_size, eps=1e-8):
        self.eps = eps
        self.vocab_size = vocab_size

    def forward(self, logits, targets):
        try:
            targets_int = targets.cpu().detach().to(torch.int64)
            ones = F.one_hot(targets_int, num_classes=self.vocab_size).float()
            for i in range(len(targets)):
                ones[i, targets_int[i]] = targets[i] - targets[i].detach() + 1.0
        except:
            print("error with targets: ", targets)
        return (
            torch.sum(-ones.cuda() @ F.log_softmax(logits + self.eps, -1).T, -1)
            .mean()
            .cuda()
        )


# def celoss(logits, targets, eps=1e-8):

# 	return torch.sum(- targets * F.log_softmax(logits+eps, -1).T, -1).mean()


# https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/model.py
class RNNDecoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size=1024,
        embed_size=1024,
        num_layers=1,
        max_seq_length=16,
    ):
        super(RNNDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # self.embed = nn.Embedding(vocab_size, embed_size)
        self.vocab_size = vocab_size + 1  # +1 for <START>
        self.embed = HotEmbedding(self.vocab_size, hidden_size)
        self.lstm = nn.LSTM(
            embed_size, hidden_size, num_layers, batch_first=True, dropout=0.5
        )
        self.max_seq_length = max_seq_length
        self.cls = nn.Sequential(
            nn.Linear(hidden_size, 20000),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(20000, vocab_size),
            nn.ReLU(),
        )

    def init_hidden(self, batch_size):
        return (
            torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda(),
            torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda(),
        )

    def forward(self, features, captions, lengths):
        batch_size = features.size(0)
        hidden = self.init_hidden(batch_size)
        start_tokens = (
            torch.ones(batch_size).cuda() * self.vocab_size
        )  # 2) pass <START> token
        captions = torch.cat((start_tokens.unsqueeze(1), captions), 1)
        embeds = self.embed(captions)  # convert caption to tokens
        lengths = [l + 1 for l in lengths]
        # print("features.shape: ",features.shape)
        # print("embeds.shape: ",embeds.shape)
        # print(captions, lengths)

        # output = torch.zeros((batch_size, captions[0].shape[1], embeds.shape[-1])) # shape: (batch_size, max_length, embed_size)
        # next_inp, hidden = self.lstm(features.unsqueeze(1), hidden) # 1) pass img feats
        # output[:,0,:], hidden = self.lstm(start_tokens, hidden)
        embeds = torch.cat(
            (features.unsqueeze(1), embeds), 1
        )  # append features to embeddings
        # print("embed, length shape :",embeds.shape, len(lengths))
        packed = pack_padded_sequence(
            embeds, lengths, batch_first=True
        )  # pack the input to skip padding backprop

        output, states = self.lstm(packed, hidden)

        output, states = pad_packed_sequence(output, batch_first=True)
        output = output[:, :-1, :]  # remove the extra term due to <START>
        # print("unpacked output: ",output.contiguous().view(-1,output.shape[-1]).shape)
        # print("caption shape: ",captions.shape)
        output = output.contiguous().view(-1, output.shape[-1])
        output = self.cls(output)
        states = tuple(state.detach() for state in states)
        return output, states

    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        states = self.init_hidden(inputs.shape[0])
        for i in range(self.max_seq_length):
            # if states != None:
            # 	print("intputs: ",inputs.shape)
            # 	for state in states:
            # 		print(state.shape)
            # else:
            # print("intputs: ",inputs.shape)
            # batch_size = inputs.size(0)
            # hidden = self.repackage_hidden(batch_size)
            # print("hidden size, ", hidden.shape)
            hiddens, states = self.lstm(
                inputs, states
            )  # hiddens: (batch_size, 1, hidden_size)
            # print("hiddens size: ",hiddens.shape)
            # states = self.repackage_hidden(states)
            outputs = self.cls(hiddens.squeeze())  # outputs:  (batch_size, vocab_size)
            if outputs.ndim == 1:
                outputs = outputs.unsqueeze(0)
            # print("outputs: ",outputs.shape)
            # print(hiddens.isinf().any(), outputs.isinf().any())
            # check = [v.data for v in self.cls.parameters()]
            # print(check)
            # _, predicted = outputs.max(1)                        # predicted: (batch_size)
            predicted = diff_argmax(outputs)
            # print("predicted: ",predicted.shape)
            # ip = predicted.detach().clone().long()
            # print(predicted.shape, ip.shape)
            # print("predicted: ",predicted.shape)
            sampled_ids.append(predicted)
            inputs = self.embed(
                predicted.detach().clone()
            )  # inputs: (batch_size, embed_size)
            # print('after embed, ',inputs.shape)
            inputs = inputs.unsqueeze(1)  # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(
            sampled_ids, 1
        )  # sampled_ids: (batch_size, max_seq_length)
        # print('sampled ids: ',sampled_ids.shape)
        return sampled_ids

    def repackage_hidden(self, h):
        """Wraps hidden states in new Tensors, to detach them from their history."""
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.repackage_hidden(v) for v in h)
        # hidden = (torch.zeros(self.num_layers, batch_size, self.hidden_size))
        return hidden


CIFAR_CLASSES = 10
CIFAR100_CLASSES = 100


class Learner(nn.Module):
    def __init__(self, enc_arch, decoder, vocab_size):
        super(Learner, self).__init__()
        self.enc_arch = enc_arch
        if enc_arch == "18":
            self.enc = resnet18(pretrained=True).cuda()
        elif enc_arch == "34":
            self.enc = resnet34().cuda()
        elif enc_arch == "50":
            self.enc = resnet50(pretrained=True).cuda()
        elif enc_arch == "101":
            self.enc = resnet101().cuda()

        self.dec = decoder.cuda().train()
        self.vocab_size = vocab_size + 1
        # if is_cifar100:
        #   teacher_h = nn.Linear(512 * self.enc.block.expansion, CIFAR100_CLASSES).cuda()
        # else:
        #   teacher_h = nn.Linear(512 * self.enc.block.expansion, CIFAR_CLASSES).cuda()
        # teacher_v = nn.Linear(512 * self.enc.block.expansion, 2).cuda()

        self.lin = nn.Linear(
            512 * self.enc.block.expansion, self.dec.hidden_size
        ).cuda()
        # if is_cifar100:

        # else:
        #     self.cls = nn.Linear(self.dec.hidden_size, CIFAR100_CLASSES).cuda()
        # self.cls = nn.Linear(self.dec.hidden_size, vocab_size).cuda()
        self.celoss = celoss(vocab_size=vocab_size)

    def forward(self, input, alphas, captions, lengths):
        feats = self.enc(input)
        # print("FEATS: ",feats)
        feats = self.lin(feats)
        # print("feats.shape: ",feats.shape)
        logits, _ = self.dec(feats, captions, lengths)
        # batch_size, seq_len = out.size[0], out.size[1]
        # print(out.view(-1,out.size(-1)).shape)
        # logits = self.cls(out.view(-1,out.size(-1)))

        return logits

    def loss(self, input, alphas, caption, length):
        # target = pack_padded_sequence(caption, length, batch_first=True)[0]
        logits = self(input, alphas, caption, length)
        # print("loss shape: ",logits.shape, caption.view(-1).shape)
        return self.celoss.forward(logits, caption.view(-1))

    def new(self):
        model_new = Learner(self.enc_arch, self.dec, self.vocab_size).cuda()
        # for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        #     x.data.copy_(y.data)
        return model_new
