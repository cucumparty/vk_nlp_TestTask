from datasets import load_dataset

def filter_russian_top_level(example):
    '''Filter top level and russian instructions

    @param example(datasets.arrow_dataset.Dataset): dataset from datasets
    '''
    return example['lang'] == 'ru' and example['parent_id'] is None

class Dataset():
    def __init__(self):
        'Initialize class Dataset'
        
        self.ds = load_dataset("OpenAssistant/oasst1")
        self.train = self.ds['train']
        self.val = self.ds['validation']

    def filter(self):
        'Filter train and validation datasets using function filter_russian_top_level'

        self.train_filtered = self.train.filter(filter_russian_top_level)
        self.val_filtered = self.val.filter(filter_russian_top_level)
    
    def print_info(self):
        'Print info about dataset'

        print(f"Отфильтрованный train: {len(self.train_filtered)} записей")
        print(f"Отфильтрованный val: {len(self.val_filtered)} записей")

        print(self.train_filtered[:1])
        