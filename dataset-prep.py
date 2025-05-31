import json
from datasets import load_dataset

datasets = load_dataset('hate-speech-portuguese/hate_speech_portuguese', split='train[:10%]')

print(datasets)

datasets.remove_columns(['hatespeech_G1', 'annotator_G1', 'hatespeech_G2', 'annotator_G2', 'hatespeech_G3', 'annotator_G3'])

datasets = datasets.train_test_split(test_size=0.2)

print(datasets[0])

datasets['train']['text']

# Removendo o \n
def removeN(example):
    example['text'] = example['text'].replace("\n", " ")
    return example

datasets = datasets.map(removeN)

# label 0 -> No Hate Speech
# label 1 -> Hate Speech
def labelChange(example):
    example['label_text'] = 'No Hate Speech' if example['label']==0 else 'Hate Speech'
    return example

datasets = datasets.map(labelChange)

datasets = datasets.remove_columns(['label'])

print(datasets['train'][0])
