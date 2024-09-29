# Custom DPO Model with Language Model Training

## Description
This project is designed to fine-tune a language model using a dataset of instructions and responses. The model can be trained to respond in a specific style, such as sarcasm. The training process leverages a dataset that includes various responses, and the evaluation is based on predefined criteria.

## Requirements
- Python 3.x
- PyTorch
- Transformers
- Other necessary libraries listed in `requirements.txt`

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```bash
   cd <project-directory>
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
To fine-tune the model and run the training process, use the following command:

```bash
python alignment.py --mode train --style <style> --train_number <train_num> --val_number <val_num> --criteria_file "criteria.txt"
```

Where:
- `--mode`: Can be `train` to start the fine-tuning process.
- `--style`: Specifies the style of response the model should focus on (e.g., "sarcasm").
- `--train_number`: Number of training samples to use.
- `--val_number`: Number of validation samples to use.
- `--criteria_file`: Path to the file containing response evaluation criteria (e.g., `criteria.txt`).

### Example:
```bash
python alignment.py --mode=train --style=сарказм --train_number=100 --val_number=10 --criteria_file=criteria.txt 
```

## Criteria File (`criteria.txt`)
The evaluation of responses is based on a criteria file. An example `criteria.txt` is as follows:

```plaintext
1 — Совсем нет сарказма, ответ прямолинейный и серьезный. 
2 — Легкий намек на сарказм, но он неявен. 
3 — Сарказм заметен, но умеренный.
4 — Сарказм явный и достаточно ощутимый.
5 — Явный сарказм, ответ наполнен язвительными комментариями и насмешками.
```

This file is used during evaluation to score the model's responses and guide training.

## Model Evaluation
After training, the model generates two responses for each input in the validation set:
- **Base Response**: The initial output from the model.
- **DPO Response**: The refined output after DPO training.

The results will be saved in an `responses.json` file.



