from keras.models import model_from_json
from main import val_data_gen,val_generator
import numpy as np


# load json and create model
json_file = open('model/model_20.json','r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into model
model1 = emotion_model.load_weights('model/model_20.weights.h5')
print('Loaded model from disk')

# pred=model1.predict(val_generator,steps=7178//64)
# final_predict=np.argmax(pred,axis=1)
# true_data=val_data_gen.classes