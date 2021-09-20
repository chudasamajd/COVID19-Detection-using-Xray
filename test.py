from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import h5py
from PIL import Image

f = h5py.File("D:/Python Projects/COVID19/vgg19.h5",'r+') 
data_p = f.attrs['training_config']
data_p = data_p.decode().replace("learning_rate","lr").encode()
f.attrs['training_config'] = data_p
f.close()

model = load_model('D:/Python Projects/COVID19/vgg19.h5')


def doOnlineInference (imagePath):
    input_image = image.load_img(imagePath, target_size=(224, 224))
    input_image_array = image.img_to_array(input_image)
    image_array_expanded = np.expand_dims(input_image_array, axis = 0)
    image_array_expanded_preprocessed = preprocess_input(image_array_expanded)
    prediction = model.predict(image_array_expanded_preprocessed)
    # outputContent = "Normal with (" + str( round( prediction[0][0]*100, 3 )) + "%) confidence.\n\n"
    # outputContent += "Coronavirus Pneumonia with ("  + str( round( prediction[0][1]*100, 3 ) ) + "%) confidence.\n\n" 
    # outputContent += "Raw neural network output array [normal,pneumonia] ~> [" + str( round( prediction[0][0], 3 )) + "," + str( round( prediction[0][1], 3 )) + "]\n\n\n"
    outputContent =  str( round( prediction[0][1]*100, 3 ) ) 
    return outputContent




##########################
# Function added by Jordan to evaluate accuracy of loaded model. Model is evaluated without the need for retraining.
# Thus, this faciliates accuracy evaluation of the saved/loaded (in 2 minutes on gtx 1060/i7 cpu) model without invocation of model-training function **model.fit**, which would take hours on the same machine.

# Evaluate saved model
from keras.preprocessing.image import ImageDataGenerator

test_sample_number = 624 #no of example ct scan images in "xray_dataset/test"

test_dataGen = ImageDataGenerator(rescale=1./255)

test_set = test_dataGen.flow_from_directory('D:/Python Projects/COVID19/chest-xray-pneumonia/chest_xray/test',
                                                target_size=(224, 224),
                                                batch_size=32,
                                                class_mode='categorical')

score = model.evaluate_generator(test_set, test_sample_number/32, workers=12)
print ("model accuracy [loss = " + str(score[0]*100) + "%, accuracy = " + str(score[1]*100) +"%]")
#See metrics using: `model.metrics_names` 
