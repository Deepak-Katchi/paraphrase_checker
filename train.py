import pandas as pd
import config
import dataset
import model

def get_dataset(path):
    df = pd.read_csv(df, sep='\t',error_bad_lines=False).dropna()
    tf_dataset = dataset.TokenisedDataset(list(df['#1 String']),
                                      list(df['#2 String']),
                                      list(df['Quality']),
                                      list(df['#1 ID'])).tokenize_data()
    return tf_dataset

    

def main():
    train_dataset = get_dataset(config.TRAINING_FILE)
    validation_dataset = get_dataset(config.VALIDATION_FILE)
    trained_model = model.BertModel().train_model(train_dataset,validation_dataset,save=True)


if __name__ == "__main__":
    main()

        
