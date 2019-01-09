from keras import models

model = models.load_model('dogs_vs_cats_small.h5')
img_path =  'D:\\all-dataset\\dogs-vs-cats-small\\test\dogs\\dog.1500.jpg'

from keras.preprocessing import image
import numpy as np

img = image.load_img(img_path,target_size=(150,150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor,axis=0)
img_tensor /=255.
import matplotlib.pyplot as plt
#plt.imshow(img_tensor[0])
#plt.show()

layer_outputs = [layer.output for layer in model.layers[:8]]
acti_model = models.Model(inputs=model.input , outputs=layer_outputs)
acti_model.summary()
actis = acti_model.predict(img_tensor)
plt.matshow(actis[7][0,:,:,3],cmap = 'viridis')
plt.show()