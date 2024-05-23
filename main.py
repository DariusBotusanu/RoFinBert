from transformers import AutoTokenizer, AutoModel

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained("racai/distilbert-base-romanian-cased")
model = AutoModel.from_pretrained("racai/distilbert-base-romanian-cased")

# tokenize a test sentence
input_ids = tokenizer.encode("Aceasta este o propoziție de test.", add_special_tokens=True, return_tensors="pt")

# run the tokens trough the model
outputs = model(input_ids)

print(outputs)


#if __name__ == "__main__":