from datasets import load_dataset
from transformers import pipeline

if __name__=='__main__':
    #Load the financial_phrasebank dataset where all labelers agree
    dataset = load_dataset("financial_phrasebank", 'sentences_allagree', trust_remote_code=True)
    
    #Make a pipeline to translate from English to Romanian
    pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-tc-big-en-ro")
    
    #Translate all sequences to Romanian
    print("Translating Sequences...")
    dataset['train']['sentence'] = pipe(dataset['train']['sentence'])
                             
    #Save the dataset to disk
    print("Saving to disk...")
    dataset.save_to_disk("datasets/ro_financial_phrasebank_all_agree.hf")
