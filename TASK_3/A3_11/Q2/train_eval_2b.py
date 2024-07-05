from evaluate import load
from transformers import T5Tokenizer, T5ForConditionalGeneration
from nltk.translate.meteor_score import single_meteor_score 

val_list_de = []
val_list_en = []

val_file = open('val.txt','r')
for line in val_file:
    temp_dict = eval(line)
    val_list_de.append(temp_dict['de'])
    val_list_en.append(temp_dict['en'])


test_list_de = []
test_list_en = []
test_file = open('test.txt','r')

for line in test_file:
    temp_dict = eval(line)
    test_list_de.append(temp_dict['de'])
    test_list_en.append(temp_dict['en'])

tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")
task_prefix = "translate English to German: "

sentences = val_list_en
inputs = tokenizer([task_prefix + sentence for sentence in sentences], return_tensors="pt", padding=True, max_length=512, truncation=True)

output_sequences = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    do_sample=False,  # disable sampling to test if batching affects output
    max_length=512
)

translated_val_de = tokenizer.batch_decode(output_sequences, max_length=512, skip_special_tokens=True)

predictions = translated_val_de
references = val_list_de

bleu = load("bleu")

bleu1_score = bleu.compute(predictions=predictions, references=references, max_order=1)
bleu2_score = bleu.compute(predictions=predictions, references=references, max_order=2)
bleu3_score = bleu.compute(predictions=predictions, references=references, max_order=3)
bleu4_score = bleu.compute(predictions=predictions, references=references, max_order=4)

print("BLEU-1 for Val Set:", bleu1_score)
print("BLEU-2 for Val Set:", bleu2_score)
print("BLEU-3 for Val Set:", bleu3_score)
print("BLEU-4 for Val Set:", bleu4_score)


meteor = []
for i in range(len(predictions)):
    temp = single_meteor_score(predictions[i].split(), references[i].split())
    meteor.append(temp)
print("AVG METEOR: ", sum(meteor)/len(meteor))


bert = load("bertscore")
bert_score = bert.compute(predictions=predictions, references=references, lang='de')
print("BERT AVERAGE of precision:", sum(bert_score['precision'])/len(bert_score['precision']))
print("BERT AVERAGE of recall:", sum(bert_score['recall'])/len(bert_score['recall']))
print("BERT AVERAGE of f1:", sum(bert_score['f1'])/len(bert_score['f1']))


sentences = test_list_en
inputs = tokenizer([task_prefix + sentence for sentence in sentences], return_tensors="pt", padding=True)

output_sequences = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    do_sample=False,  # disable sampling to test if batching affects output
)

translated_test_de = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)

predictions = translated_test_de
references = test_list_de
bleu = load("bleu")

bleu1_score = bleu.compute(predictions=predictions, references=references, max_order=1)
bleu2_score = bleu.compute(predictions=predictions, references=references, max_order=2)
bleu3_score = bleu.compute(predictions=predictions, references=references, max_order=3)
bleu4_score = bleu.compute(predictions=predictions, references=references, max_order=4)

print("BLEU-1 for Test Set: ", bleu1_score)
print("BLEU-2 for Test Set: ", bleu2_score)
print("BLEU-3 for Test Set: ", bleu3_score)
print("BLEU-4 for Test Set: ", bleu4_score)

meteor = []
for i in range(len(predictions)):
    temp = single_meteor_score(predictions[i].split(), references[i].split())
    meteor.append(temp)
print("AVG METEOR: ", sum(meteor)/len(meteor))


bert = load("bertscore")
bert_score = bert.compute(predictions=predictions, references=references, lang='de')
print("BERT AVERAGE of precision:", sum(bert_score['precision'])/len(bert_score['precision']))
print("BERT AVERAGE of recall:", sum(bert_score['recall'])/len(bert_score['recall']))
print("BERT AVERAGE of f1:", sum(bert_score['f1'])/len(bert_score['f1']))

