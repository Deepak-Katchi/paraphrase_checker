import config
import tensorflow as tf
import transformers

class BertModel():
    def __init__(self):
        self.model = transformers.TFBertForSequenceClassification.from_pretrained(config.MODEL_TYPE)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

    def compile_model(self):
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=[self.metric])

    def train_model(self,train_dataset,validation_dataset,save=False):
        self.get_compiled_bert_model()
        train_data = train_dataset.shuffle(100).batch(config.TRAIN_BATCH_SIZE).repeat(2)
        valid_data = validation_dataset.batch(config.VALID_BATCH_SIZE)
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)
        self.model.fit(train_data,epochs=config.EPOCHS, batch_size=config.TRAIN_BATCH_SIZE,validation_data=valid_data,callbacks=[callback])
        if save:
            self.model.save_weights(config.MODEL_WEIGHTS_PATH)
        return self.model

    
