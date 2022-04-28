from torch.utils.data import dataloader, Dataset

from masked_cross_entropy import sequence_mask


class DualDataset(Dataset):
    def __init__(self, equations, numberlists, labels):
        self.equations = equations
        self.numberlists = numberlists
        self.labels = labels
        self.operator = ['+', '-', '*', '/', '**', '^']
        self.number_vocabulary = ['UNK', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', '(', '/', ')', '%']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index]
        dataset = [self.equations[index], self.numberlists[index]]
        sample = {'text': dataset, 'label': label}
        return sample
