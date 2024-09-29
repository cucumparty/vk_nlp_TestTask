import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def print_func(*args):
    for arg in args:
        print( arg)
        print('--------------------------')
    print("==============================")


class Model():
    def __init__(self, MODEL_NAME, PROMPT, MODE, STYLE, CRITERIA):
        '''Initialize class Model

        @param MODEL_NAME(str): name of the model
        @param PROMPT(str): prompt for the model
        @param MODE(str): 'train' or 'test'
        @param STYLE(str): style of answers
        @param CRITERIA(path): path to criteria.txt file
        '''
        self.prompt = PROMPT
        self.mode = MODE
        self.style = STYLE
        self.criteria = CRITERIA
        self.lr = 5e-5
        self.batch_size = 2

        self.bnb_config = BitsAndBytesConfig(
            llm_int8_enable_fp32_cpu_offload = True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=self.bnb_config,
            torch_dtype=torch.bfloat16
        )

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        

    def generate_response(self, prompt, inputs):
        '''Generate response to instruction
        
        @param prompt(str): prompt to model
        @param inputs(str): instruction
        '''
        prompt = self.tokenizer.apply_chat_template([{
            "role": "system",
            "content": prompt
        }, {
            "role": "user",
            "content": inputs
        }], tokenize=False, add_generation_prompt=True)
        data = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        data = {k: v.to(self.model.device) for k, v in data.items()}
        output_ids = self.model.generate(**data, max_new_tokens=200)[0]
        output_ids = output_ids[len(data["input_ids"][0]):]
        output = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        return output
        

    def get_responses_and_scores(self, train_filtered):
        '''Generate 2 answers to train_filtered, get scores to this answers, make dataset for DPO

        @param train_filtered(datasets.arrow_dataset.Dataset): filtered dataset of instructions
        '''
        print('GENERATE')
        self.model.eval()
        self.data_for_dpo = []

        for num, instruction in enumerate(train_filtered['text']):
            print(f'Instruction {num}')

            with torch.no_grad():
                response_1 = self.generate_response(self.prompt, instruction)
                response_2 = self.generate_response(self.prompt, instruction)

                prompt = f"Ты — русскоязычный автоматический ассистент. Ты даешь оценку ответов на инструкции по критерию - {self.style}"

                evaluation_prompt_1 = f"Оцени этот ответ на {self.style}: \"{response_1}\". Пожалуйста, дай оценку от 1 до 5. Где {self.criteria}"
                score_1 = self.generate_response(prompt, evaluation_prompt_1)
                evaluation_prompt_2 = f"Оцени этот ответ на {self.style}: \"{response_2}\". Пожалуйста, дай оценку от 1 до 5. Где {self.criteria}"
                score_2 = self.generate_response(prompt, evaluation_prompt_2)

            if self.mode == 'train':
                print_func('Response 1:',instruction, response_1, score_1)
                print_func('Response 2:',instruction, response_2, score_2)

            score_1 = [int(s) for s in score_1 if s.isdigit()]
            score_2 = [int(s) for s in score_2 if s.isdigit()]

            self.data_for_dpo.append({
                "instruction": instruction,
                "response_1": response_1,
                "score_1": int(score_1[0]),
                "response_2": response_2,
                "score_2": int(score_2[0])
            }) 
        
        return self.data_for_dpo


    def get_responses(self, val_filtered): 
        '''Generate responses to val_filtered
        
        @param val_filtered(datasets.arrow_dataset.Dataset): filtered dataset of instructions
        '''           
        responses = []
        for num, instruction in enumerate(val_filtered['text']):  
            print(f'Instruction {num}')
            response = self.generate_response(self.prompt, instruction)
            responses.append(response)

            if self.mode == 'train':
                print_func(instruction, response)

        return responses
    

    def dpo_training(self):
        'Train model using DPO'

        print('DPO_TRAINING')
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
 
        batch_size = self.batch_size
        num_batches = len(self.data_for_dpo) // batch_size + (len(self.data_for_dpo) % batch_size > 0)

        for batch_idx in range(num_batches):
            optimizer.zero_grad()
            total_loss = torch.zeros(1, requires_grad=False)

            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            batch = self.data_for_dpo[start_idx:end_idx]  

            for item in batch:
                instruction = item["instruction"]
                response_1 = item["response_1"] 
                score_1 = item["score_1"]        
                response_2 = item["response_2"]  
                score_2 = item["score_2"]        

                if score_1 > score_2:
                    preferred_response = response_1
                    non_preferred_response = response_2
                else:
                    preferred_response = response_2
                    non_preferred_response = response_1

                prompt = f"Инструкция: {instruction}\nОтвет 1: {preferred_response}\nОтвет 2: {non_preferred_response}\n"
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)

                outputs = self.model(**inputs, labels=inputs.input_ids)
                loss = outputs.loss
                total_loss += loss

            print(f'Средний loss в батче № {batch_idx}: {total_loss / len(batch)}')

            if total_loss > 0:  
                total_loss.backward()
                optimizer.step()

        print("Модель успешно дообучена с использованием DPO.")