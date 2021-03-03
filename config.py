import transformers

MAX_LEN = 128
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 16
EPOCHS = 5
MODEL_TYPE = "bert-base-uncased"
TRAINING_FILE = "/AITutor/MRPC_Tutorial/glue_data/MRPC/msr_paraphrase_train.txt"
VALIDATION_FILE = "/AITutor/MRPC_Tutorial/glue_data/MRPC/msr_paraphrase_test.txt"
MODEL_WEIGHTS_PATH = "bert_model_weights/"
TOKENIZER = transformers.BertTokenizer.from_pretrained(MODEL_TYPE, do_lower_case=True)
