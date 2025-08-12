from datasets import load_dataset

def get_dataset():
    path2DS_SQUAD2 = '../clear_docs/squad_cache'
    dataSet = load_dataset("squad_v2", cache_dir=path2DS_SQUAD2)
    return dataSet