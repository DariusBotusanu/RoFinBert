from datasets import load_dataset
from transformers import pipeline
import concurrent.futures
from tqdm import tqdm

def translate_sentences(sentences=None, translation_pipe=None, file_path='./datasets/all_agree_ro.txt'):
    '''
    Translates sentences from a dataset. Progress is tracked in a file and can be resumed if the program stops.
    '''
    with open(file_path, 'r', encoding="utf-8") as f:
        current_line = len(f.readlines())
        
    with open(file_path, 'a+', encoding="utf-8") as f:
        for sentence in tqdm(sentences[current_line:]):
            translated_sentence = translation_pipe(sentence)[0]['translation_text']
            f.write(translated_sentence)
            f.write('\n')
            f.flush()

def load_and_translate(dataset_name, file_path):
    dataset = load_dataset("financial_phrasebank", dataset_name, trust_remote_code=True)
    translate_sentences(sentences=dataset['train']['sentence'], translation_pipe=pipe, file_path=file_path)


if __name__=="__main__":
    #Make a pipeline to translate from English to Romanian
    pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-tc-big-en-ro")
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        futures.append(executor.submit(load_and_translate, 'sentences_allagree', './datasets/all_agree_ro.txt'))
        futures.append(executor.submit(load_and_translate, 'sentences_75agree', './datasets/75_agree_ro.txt'))

    for future in concurrent.futures.as_completed(futures):
        print(f"Task Completed: {future.result()}")