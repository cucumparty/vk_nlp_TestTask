from dataset import Dataset
from model import Model
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--style', help='Choose style of answers')  
parser.add_argument('-m', '--mode', help = 'Choose train or test')  
parser.add_argument('-t', '--train_number', help = 'Choose number of train instructions to use to train model using DPO')  
parser.add_argument('-v', '--val_number', help = 'Choose number of validation instructions to use to compare base model and model using DPO') 
parser.add_argument('-с', '--criteria_file', help = 'Write path to file with criteria for style') 

args = parser.parse_args()
MODE = args.mode
STYLE = args.style
CRITERIA = open (args.criteria_file, 'r').read()

MODEL_NAME = "IlyaGusev/saiga_llama3_8b"
DEFAULT_SYSTEM_PROMPT = f"Ты — русскоязычный автоматический ассистент. Ты разговариваешь с людьми,помогаешь им и даешь ответы на их вопросы. Стиль твоих ответов - {STYLE}"

dataset = Dataset()
dataset.filter()
train = dataset.train_filtered
val = dataset.val_filtered

if args.train_number:
    train = train[:int(args.train_number)]
if args.val_number:    
    val = val[:int(args.val_number)]

model = Model(MODEL_NAME, DEFAULT_SYSTEM_PROMPT, MODE, STYLE, CRITERIA)

model.get_responses_and_scores(train)
print('BASE VAL')
base_responses = model.get_responses(val)
model.dpo_training()
print('DPO VAL')
dpo_responses = model.get_responses(val)
val['base_response'] = base_responses
val['dpo_response'] = dpo_responses

with open('responses.json', 'w', encoding='utf-8') as f:   
    json.dump(val, f, ensure_ascii=False, indent='')