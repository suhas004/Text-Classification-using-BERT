# Code developed by @Suhas Sasetty 

from ktrain import text
import pandas as pd
import ktrain
from tensorflow.keras.models import load_model


#model= load_model('tf_model.h5')

class bert_predict:

    def __init__(self,path,text):
        self.predictor = ktrain.load_predictor(path)
        self.text=text

    def predict_sentiment(self):

        prediction=self.predictor.predict(self.text)

        return prediction

if __name__ == '__main__':
    
    path = 'weights/'    #specify your trained weights path here 
    text = 'So, why does this show sucks'  #give your input text here
    
    model=bert_predict(path,text)
    
    output = model.predict_sentiment()

    print(output)