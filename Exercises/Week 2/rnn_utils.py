import numpy as np
from collections import Counter
import datasets
from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader


class RNNDataset(Dataset):
    # read more about custom datasets at https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    def __init__(self,
                 train_dataset: datasets.arrow_dataset.Dataset,
                 test_dataset: datasets.arrow_dataset.Dataset,
                 vocabulary_size: int
                ):
        self.prepared_train_dataset = self.prepare_rnn_lm_dataset(train_dataset)
        self.prepared_test_dataset = self.prepare_rnn_lm_dataset(test_dataset)
        self.max_seq_len = self.get_max_seq_len() # everything will be padded to this length in this dataloder
        
        dataset_vocab = self.get_dataset_vocabulary(train_dataset)
        # defining a dictionary that simply maps tokens to their respective index in the embedding matrix
        self.word_to_index = {word: idx for idx,word in enumerate(dataset_vocab)}
        self.index_to_word = {idx: word for idx,word in enumerate(dataset_vocab)}
    
        self.pad_idx = self.word_to_index["<pad>"]
        
    def __len__(self):
        return len(self.prepared_train_dataset)
    
    def get_max_seq_len(self):
        max_seq_len_train = max([len(sample.split()) for sample in self.prepared_train_dataset])
        max_seq_len_test = max([len(sample.split()) for sample in self.prepared_test_dataset])
        return max(max_seq_len_train, max_seq_len_test)
    
    def get_encoded_test_samples(self):
        all_token_lists = [sample.split() for sample in self.prepared_test_dataset]
        max_seq_len = max([len(sample_tokens) for sample_tokens in all_token_lists])
        # padding every sentence to max_seq_length
        all_token_ids = [[self.word_to_index.get(word, self.word_to_index["<unk>"])
                          for word in token_list] + [self.pad_idx] * (self.max_seq_len - len(token_list))
                         for token_list in all_token_lists
                        ]

        return torch.tensor(all_token_ids)
    
    def __getitem__(self, idx):
        # here we need to transform the data to the format we expect at the model input
        token_list = self.prepared_train_dataset[idx].split()
        # having a fallback to <unk> token if an unseen word is encoded.
        token_ids = [self.word_to_index.get(word, self.word_to_index["<unk>"]) for word in token_list]
        return torch.tensor(token_ids + [self.pad_idx] * (self.max_seq_len - len(token_list)))
    
    def decode_idx_to_word(self, token_id):
        return [self.index_to_word[id_.item()] for id_ in token_id]
    
    def get_dataset_vocabulary(self, train_dataset: datasets.arrow_dataset.Dataset):
        vocab = sorted(set(" ".join([sample["sentence"] for sample in train_dataset]).split()))
        # we also add a <start> token to include initial tokens in the sentences in the dataset
        vocab += ["<start>", "<pad>"]
        return vocab
    
    @staticmethod
    def prepare_rnn_lm_dataset(target_dataset: datasets.arrow_dataset.Dataset):
        '''
        a "<start>" token has to be added before every sentence
        args:
            target_dataset: the target dataset where its consecutive tokens of length 'window_size' should be extracted
            window_size: the window size for the language model
        output:
            prepared_dataset: a list of strings each containing 'window_size' tokens.
        '''
        prepared_dataset = []
        for sample in target_dataset:
            prepared_dataset.append(f"<start> {sample['sentence']}")
        
        return prepared_dataset
    
    
class RNN_language_model(torch.nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, dropout=0.001, pad_idx = -1):
        super().__init__()

        self.hidden_dim = hidden_dim
        
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index = pad_idx)
        self.embedding = torch.nn.Embedding(vocab_size, emb_dim) # word embeddings
        self.rnn = torch.nn.RNN(emb_dim, hidden_dim, num_layers=4)
        self.dropout = torch.nn.Dropout(dropout)
        self.lm_decoder = torch.nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, context):
        context = context.t() # transposing it for RNN model
        #context = [src len, batch size]
        
        embedded = self.dropout(self.embedding(context))
        
        #embedded = [src len, batch size, emb dim]
        
        outputs, hidden = self.rnn(embedded)
        #outputs = [src len, batch size, hidden_dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        outputs = self.lm_decoder(outputs.permute(1, 0, 2))[:, :-1, :].permute(0, 2, 1)
        target_tokens = context.t()[:, 1:]
        loss = self.criterion(input=outputs, target=target_tokens)
        return loss
    