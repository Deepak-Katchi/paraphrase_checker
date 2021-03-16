from fastapi import FastAPI
import tensorflow as tf
import transformers
from pydantic import BaseModel
import uvicorn

import config
from model import BertModel

class Sentence(BaseModel):
    sentence_1: str
    sentence_2: str


def get_tokenizer(model_type):
    return transformers.BertTokenizer.from_pretrained(model_type, do_lower_case=True)

def get_compiled_bert_model(model_type):
    
    model = transformers.TFBertForSequenceClassification.from_pretrained(model_type)
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    return model

tokenizer = config.TOKENIZER
bert = BertModel()
bert.compile_model()
model = bert.model
model.load_weights(config.MODEL_WEIGHTS_PATH)

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}
    

@app.post("/predict/")
def predict(s_obj: Sentence):
    

    
    output = tf.nn.softmax(model(tokenizer(s_obj.sentence_1,s_obj.sentence_2,max_length=128,return_tensors='tf'))).numpy()[0]
    print(output)
    
    return {"Paraphrase_Probability " : str(output[0][1]) }
    
    

    
if __name__ == "__main__":

    # Run app with uvicorn with port and host specified. Host needed for docker port mapping

    uvicorn.run(app, port=8000, host="0.0.0.0")    
    
    
