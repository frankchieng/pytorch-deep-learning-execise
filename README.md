# pytorch-deep-learning-exercise
implementing the exercise work of https://github.com/mrdbourke/pytorch-deep-learning

datasets are under data directory

i highly recommend anyone who want to learn pytorch should learn about this author's tutorial and finish the exercise work,happy coding

tips:
i've modified the default tiny VGG model achitecture in 03_pytorch_computer_vision_exercise_solutions.ipynb, add BatchNorm2D and Dropout,just compare it with the orginal model,it turns out the new model generalized well on the test data,cuz solve the overfitting problem a little bit.

in 04_pytorch_custom_datasets.ipynb exercise,after increase the dataset of Food101 including steak,sushi and pizza to 20% of total images, train with hidden_units=20 TinyVGG model(including BatchNorm2D and Dropout),1000 epochs,the model performs better to classify multiclass food which the model has not learned when inference with new images i download from Google,you can give it a shot with more epochs.

in 05_pytorch_going_modular_script_mode.ipynb for implementing and adding the command line parameters of training process, i just create another script file called train_cli_params.py, models can be saved and reload thr pytorch for prediction in predict.py

in 06_pytorch_transfer_learning.ipynb you can fine-tuning the pre-trained models in torchvision.models like efficientnet_b0 and efficientnet_b2 with the last classified layer hyperparameters,it turns out b0 performs much better than b2 model,so when you choose a pre-trained model,you should choose X0,not like X1,X2 etc..

you can contact me on weixin in mainland of China：

![image](https://github.com/frankchieng/imagegeneration/blob/main/wechat.jpg)
