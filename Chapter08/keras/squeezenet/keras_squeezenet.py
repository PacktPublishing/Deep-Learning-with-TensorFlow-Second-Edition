import os
import numpy as np
import squeezenet as sq
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing import image


model = sq.SqueezeNet()
img = image.load_img('squeeze_test.jpg', target_size=(227, 227))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
print('Predicted:', decode_predictions(preds))


