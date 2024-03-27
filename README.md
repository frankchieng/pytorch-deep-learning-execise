# pytorch-deep-learning-exercise
implementing the exercise work of https://github.com/mrdbourke/pytorch-deep-learning

datasets are under data directory, food101 is a big dataset around 4-5G, so i download the original dataset under data folder, and split the image to train and test folders with 04_custom_data_creation.ipynb

i highly recommend anyone who want to learn pytorch should learn about this author's tutorial and finish the exercise work,happy coding

tips:
i've modified the default tiny VGG model achitecture in 03_pytorch_computer_vision_exercise_solutions.ipynb, add BatchNorm2D and Dropout,just compare it with the orginal model,it turns out the new model generalized well on the test data,cuz solve the overfitting problem a little bit.

in 04_pytorch_custom_datasets.ipynb exercise,after increase the dataset of Food101 including steak,sushi and pizza to 20% of total images, train with hidden_units=20 TinyVGG model(including BatchNorm2D and Dropout),1000 epochs,the model performs better to classify multiclass food which the model has not learned when inference with new images i download from Google,you can give it a shot with more epochs.

in 05_pytorch_going_modular_script_mode.ipynb for implementing and adding the command line parameters of training process, i just create another script file called train_cli_params.py, models can be saved and reload thr pytorch for prediction in predict.py

in 06_pytorch_transfer_learning.ipynb you can fine-tuning the pre-trained models in torchvision.models like efficientnet_b0 and efficientnet_b2 with the last classified layer hyperparameters,after 10 epochs training,it turns out the EfficientNet_B2 convergence faster and stable than EfficientNet_B0,as a conclusion,like EfficientNet_B7 model,the model with more parameters and larger of size, performs much better than the smaller model

efficientnet_b0 model trained after 10 epochs accuracy:
![efficient_bo_accu_10epochs](https://github.com/frankchieng/pytorch-deep-learning-execise/assets/130369523/585502f1-dd9a-46ed-9af2-b985293fd515)

the same training datasets and 10 epochs of efficientnet_b2 model accuracy:
![efficient_b2_accu_10epochs](https://github.com/frankchieng/pytorch-deep-learning-execise/assets/130369523/70abcc82-663c-469d-9577-3f229f5dc01a)

in 07_pytorch_experiment_tracking.ipynb i trained the whole food101 datasets classification downstream task with EfficientNet_B7 model under GTX 4090 GPU,it took around 11 mins, so if you want to optimized and gain more precise accuracy,you can increase the training epochs and it will take longer,anyway,it's so interesting to test with different raw data and finetuning,you should give a try.

![截图 2024-03-22 18-06-49](https://github.com/frankchieng/pytorch-deep-learning-execise/assets/130369523/29fed6f4-47f7-41bb-b1e4-38c387da0d4b)

[vision transformer research paper](https://github.com/frankchieng/pytorch-deep-learning-execise/blob/main/vision_transformer.pdf) uploaded,you should read and learn end-to-end, the self-attention formula as below in the domain-specific of AI is the new holy grail as E=mc^2 of mass-energy equivalence

![image](https://github.com/frankchieng/pytorch-deep-learning-execise/assets/130369523/dc5d4e48-bc6f-4168-951c-751b91225609)

in 08_pytorch_paper_replicating.ipynb create a vision transformer architecture model from scratch, the vit_b_16 pre-trained model which built on vision transformer's accuracy performance higher than EfficientNet_B7 model,and the model size is about 10-11 times bigger than EfficientNet_B2,1.3 times bigger than EfficientNet_B2, so it's a trade-off you should make if you wanna a better performance or faster loading inference when you deploy online.

vit_b_16 pre-trained model downstream task training metrics as below:
![image](https://github.com/frankchieng/pytorch-deep-learning-execise/assets/130369523/0b7a7aa9-b8a6-4df1-8ebe-e1fd4f0a8b9a)

the vit_b_16 model classification accuracy of unseen image is much more deterministic
![image](https://github.com/frankchieng/pytorch-deep-learning-execise/assets/130369523/438d03f8-2597-4adb-bfa3-73018f3afb34)

in 09_pytorch_model_deployment.ipynb,when you upload your trained model in gradio to huggingface space,some modifications have been made:First,when you git push your files,you should use the huggingface Access Tokens instead of your huggingface account password. Second,when you encount git push binary files like images or video files with this command(git lfs track "your_file" # or *.your_extension),if it failed,you should run git filter-branch --tree-filter 'rm -rf path/to/your/file' HEAD, then git push origin --force --all
it's obvious the ViT architecture model outperform the efficient model in the end, so you should do a trade-off when you deploy a model with accuracy and inference speed.

ViT-B/16 fine-tuning model with food101 around 100 thousand full-datasets 10 epochs training result:
![image](https://github.com/frankchieng/pytorch-deep-learning-execise/assets/130369523/0ae29516-d75c-461c-b5f3-660f91a009e3)

you can check out my deployment [mini model of food101](https://huggingface.co/spaces/frank-chieng/foodvision_mini) or [big model of food101](https://huggingface.co/spaces/frank-chieng/foodvision_big)

in A_Quick_PyTorch2.0_Tutorial.ipynb training and testing with pytorch2.x new feature torch.compile(model),especially the first epoch takes more time than the non-compile model(for warm up of optimization),the rest epochs will be less time compared to non-compile one,so if you training for longer and take the full advantage of your GPU metrics, like with more batch_size,more training datasets(in case of not OOM),the overall time will be shorter
![image](https://github.com/frankchieng/pytorch-deep-learning-execise/assets/130369523/13556aed-9599-4666-8ae1-1ddd32812a22)

Markdown exmaples(include Latex syntax):
# Header1
## Header2
### Header3

text

*italics*

**bold**


$$
z =:\sum_{i=1}^{n} w_i x_i
$$

$$ \sqrt{ \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^2 } $$

$$ p_i = \text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}} $$

[Pytorch 2.0 tutorial](https://www.learnpytorch.io/pytorch_2_intro/)

> quote text

> [What are embeddings?](https://github.com/veekaybee/what_are_embeddings/)

you can contact me on weixin in mainland of China：

![image](https://github.com/frankchieng/imagegeneration/blob/main/wechat.jpg)
