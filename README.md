# Distracted Driver Classifer
Training a CNN to classify images of distracted drivers into various classes: Texting, Calling, Reaching Back, etc  
Data obtained from Kaggle Dataset: https://www.kaggle.com/c/state-farm-distracted-driver-detection

Used transfer learning (i.e. using ResNet50 because it's one of the best models out there IMO) to build the model.  
I didn't use the test set provided because I was too lazy to write a script that would read values from the CSV file provided (although that would be the next logical step)  
Instead, I manually moved 200 photos from each category into a separate validation set forming a total "test" set of 1,600 photos  

Best accuracy so far: 89.8%

Took me 4 experiments to achieve the accuracy above.

# Learning points:
- If you are facing overfitting, one of the best things you can do is to simply unfreeze more layers. Overfitting means the model is unable to generalize well to new examples, so adding more parameters to the model will make it more powerful
- USE DATA AUGMENTATION CAREFULLY. One of the mistakes I made when I first trained the model was that I was applying shearing, zoom, during the data augmentation process. Zooming in to the picture in particular resulted in losing key information like the hands of the drivers were missing. For instance, in some pictures, the driver was texting, but because of the zooming in effect, that key information was lost. Removing these data augmentation improved the performance of my model.
- Dropout helps, but to a lesser extent. I used a high dropout parameter of 0.9 in this case. 
- Having both Dropout and L2 regularization in place didn't help, it actually worsened the performance of the model. Why that is the case, I'll leave to the academics.
- It's really helpful to plot the graph of the training loss and accuracy to visualize your training better.

# Next steps:
- Figure out why my tensorboard isn't working.
- Have a proper test set, where the ground truth values will be read directly from the csv file (probably using pandas or something)
