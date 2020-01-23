import torch as _torch
import torch.nn as _nn


class _CoverageNetwork(_nn.Module):

    def __init__(self):
        super(_CoverageNetwork, self).__init__()
        self.features = _nn.Sequential(
            _nn.Upsample(scale_factor=1, mode='nearest'),
            _nn.Upsample(scale_factor=1, mode='bilinear'),
            _nn.Upsample(scale_factor=1, mode='bicubic'),
            _nn.Conv2d(3, 64, kernel_size=7), _nn.LeakyReLU(inplace=True),
            _nn.BatchNorm2d(64))
        self.avgpool = _nn.AvgPool2d(kernel_size=7)
        self.classifier = _nn.Sequential(_nn.Dropout(), _nn.Linear(576, 10),
                                         _nn.Softmax(-1))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = _torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class _RecursiveNetwork(_nn.Module):

    def __init__(self):
        super(_RecursiveNetwork, self).__init__()
        self.l1 = _nn.Linear(10, 10)

    def forward(self, x):
        x = self.l1(x)
        x = self.l1(x)
        x = self.l1(x)
        return x


class _MultipleInputNetwork(_nn.Module):

    def __init__(self):
        super(_MultipleInputNetwork, self).__init__()
        self.conv = _nn.Conv2d(3, 16, 3)

    def forward(self, inp1, inp2):
        inp = inp1 * inp2
        out = self.conv(inp)
        return out


class _SequenceNetwork(_nn.Module):

    def __init__(self, mode='lstm'):
        super(_SequenceNetwork, self).__init__()
        seq_obj = {'lstm': _nn.LSTM, 'gru': _nn.GRU, 'rnn': _nn.RNN}
        self.embedding = _nn.Embedding(15, 32)
        self.seq = seq_obj[mode](32, 64)
        self.linear = _nn.Linear(64, 5)

    def forward(self, sentences, lens):
        x = _nn.utils.rnn.pad_sequence(sentences)
        x = self.embedding(x)
        x = _nn.utils.rnn.pack_padded_sequence(x, lens)
        x, _ = self.seq(x)
        x, _ = _nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = self.linear(x)
        return x


class _DummyDataset(_torch.utils.data.Dataset):

    def __init__(self):
        self.data = _torch.randn(100, 3, 24, 24) + 10000

    def __getitem__(self, index):
        x = self.data[index]
        return x, _torch.Tensor(1)

    def __len__(self):
        return len(self.data)
