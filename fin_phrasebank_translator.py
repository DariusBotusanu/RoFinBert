from datasets import load_dataset
from transformers import pipeline
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
        
        
if __name__=="__main__":
    #Load the financial_phrasebank dataset where all labelers agree
    dataset = load_dataset("financial_phrasebank", 'sentences_allagree', trust_remote_code=True)

    #Make a pipeline to translate from English to Romanian
    pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-tc-big-en-ro")

    translate_sentences(sentences=dataset['test']['sentence'], translation_pipe=pipe, file_path='./datasets/test_all_agree_ro.txt')
    
##Studiaza pe ultimul strat embeddingurile
##Set de embeddings pentru cuvinte specifice domeniului
##TFIDF, comparare intre clasificari
