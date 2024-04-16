from datasets import load_dataset
from transformers import pipeline
from tqdm import tqdm

def translate_save(sentence, translation_pipe=None, file_path='./datasets/all_agree_ro.txt'):
    translated_sentence = translation_pipe(sentence)[0]['translation_text']
    with open(file_path, 'a+', encoding="utf-8") as f:
        f.write(translated_sentence)
        f.write('\n')


if __name__=="__main__":
    #Load the financial_phrasebank dataset where all labelers agree
    dataset = load_dataset("financial_phrasebank", 'sentences_allagree', trust_remote_code=True)

    #Make a pipeline to translate from English to Romanian
    pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-tc-big-en-ro")

    for sentence in tqdm(dataset['train']['sentence']):
        translate_save(sentence, translation_pipe=pipe)

    # #Translate all sequences to Romanian
    # print("Translating Sequences...")
    # dataset['train']['sentence'] = pipe(dataset['train']['sentence'])
    #                          
    # #Save the dataset to disk
    # print("Saving to disk...")
    # dataset.save_to_disk("datasets/ro_financial_phrasebank_all_agree.hf")
