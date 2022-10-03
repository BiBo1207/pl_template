from torch.utils.data import Dataset


class TestDataNoGT(Dataset):
    def __init__(self, hparams):
        super(TestDataNoGT, self).__init__()
        self._root = hparams['data']['test']['dir']
        self._size = hparams['data']['test']['img_size']

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass
