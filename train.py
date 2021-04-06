
# Code developed by @Suhas Sasetty


import numpy as np
#import tensorflow as tf
import ktrain
from ktrain import text
import pandas as pd

class bert:

    def __init__(self,train_data,test_data):

        self.data_train=pd.read_excel(train_data,dtype=str)
        self.data_test=pd.read_excel(test_data,dtype=str)


    def preprocess(self):


        (X_train, y_train), (X_test, y_test), preprocess = text.texts_from_df(train_df=self.data_train,
                                                                       text_column="Reviews",
                                                                       label_columns="Sentiment",
                                                                       maxlen=500,
                                                                       val_df=self.data_test,
                                                                       preprocess_mode='bert')
        model = text.text_classifier(name='bert',
                             train_data=(X_train, y_train),
                             preproc=preprocess)

        learner = ktrain.get_learner(model=model,
                             train_data=(X_train, y_train),
                             val_data=(X_test, y_test),
                             batch_size=6)


        return X_train,y_train,X_test,y_test,preprocess,learner,model

    def train(self,X_train, y_train, X_test, y_test, model): # Using GPU(P100) it will take around 24-35 hours for training the model

        learner = ktrain.get_learner(model=model,
                                     train_data=(X_train, y_train),
                                     val_data=(X_test, y_test),
                                     batch_size=6)
        learner.autofit(lr=2e-5, epochs=3)

        #learner.save_model('tf_model.h5')

        return learner



    def predictmodel(self,learner.model,preprocess,text):
        data = text
        learner =learner.model
        predictor = ktrain.get_predictor(learner,preprocess)
        print(predictor.predict(data))





if __name__ == "__main__":


    train_data = 'data/train.xlsx'
    test_data = 'data/test.xlsx'


    model = bert(train_data,test_data)

    X_train, y_train, X_test, y_test, preprocess, learner, model = model.preprocess()

    learner = model.train(X_train, y_train, X_test, y_test,model)

    #predicting the sentiment

    text = 'So why does this show suck'
    model.predictmodel(learner.model,preprocess,text)













