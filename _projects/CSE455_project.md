---
layout: page
title: Birds Birds Birds
description: My final project for CSE 455 Kaggle Competition
img: assets/img/cse_455.jpg
importance: 1
category: Coursework
---

This is my final project for CSE 455, UW CSE's Computer Vision course taught by Joseph Redmon.

The goal of this competition is to use the bird images/data provided in order to train a machine learning model
of choice and to score as high as possible on the test set of bird images. For my model choice, I chose to use
a ResNet-18 model, as these are usually the most common (along with tons of ensembling) when considering the top
of the leaderboards for many Computer Vision related tasks.

<h1>My General Approach</h1>
Similar to most Kaggle Computer Vision competitions, the people who win usually use a ton of GPUs
and ensemble like 10 models in order achieve the most extreme level of accuracy. Unfortunately, I am
just an undergraduate and I do not have that much money nor the resources to do so much compute in so little time.

So here was my approach:
<dl>
  <dt>1. Data Pre-processing</dt>
  <dd>- Thankfully, most of the data is already prepared in terms of resizing and correct labels.</dd>
  <dd>- This means that the majority of my focus can go towards specific transforms to the data in order to augment and make the model more robust to different types of images of the same bird.</dd>
  <dt>2. Model and hyperparameters</dt>
  <dd>- This will probably the part where I spend the most time due to the importance of hyperparameters in performance, especially accuracy.</dd>
  <dd>- I am pretty sure <em>ResNet-18</em> will be enough, but what I don't know if how well just one fully-connected layer on top will do. I may have to increase or get creative here.</dd>
  <dt>3. Training Loop Code</dt>
  <dd>- There is some code posted on the CSE 455 website (on a ipynb) that looks useful, but I will most likely stick to the PyTorch documentation in order to achieve my results.</dd>
</dl>

<h1>Walkthrough of my approach</h1>
First, I start by making sure that I have all the data necessary: <br>
<h2>imports</h2>
{% highlight  linenos %}
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import glob
import pandas as pd
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
{% endhighlight %}

<br>
<h2>sanity check</h2>
{% highlight  linenos %}
# Checking to make sure I have the correct data and am not going insane
DIR_PATH = "/kaggle/input/birds23wi/birds/"
labels_df = pd.read_csv(DIR_PATH + 'labels.csv')
names_df = pd.read_table(DIR_PATH + 'names.txt')
print(f"Num training samples: {len(labels_df['class'])}")
print(f"Total number of classes: {len(labels_df['class'].unique())}")
print(f"Total number of unique names in the text file for mapping labels: {len(names_df)}") # Not sure why there is a class missing, but this works out later
{% endhighlight %}

<br>
<h2>Model Choice and Hyperparameters</h2>
The outputs here look fine and there isn't anything obviously wrong. Then, I moved onto setting up my hyperparameters
and training loop:
{% highlight  linenos %}
data_dir = "./data/hymenoptera_data"

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "resnet"

# Number of classes in the dataset
num_classes = 555

# Batch size for training (change depending on how much memory you have)
batch_size = 64

# Number of epochs to train for
num_epochs = 30

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = True
{% endhighlight %}
<br>

<h2>Training Loop</h2>
Here is the PyTorch training loop:
{% highlight  linenos %}
# Creating the PyTorch Training Loop
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history
{% endhighlight %}
<br>
<h3>Freezing all but the very top layer</h3>
I was curious to see how far I could get with terms of accuracy <em><b>with only one linear layer on top of the ResNet-18 model</b></em>
with everythin else frozen.
{% highlight  linenos%}
# Freezing all the weights of the model that gets passed into this function
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
{% endhighlight %}
<br>
<h3>Implementing Data Augmentation according to PyTorch Docs</h3>
I think this is possibly the most important part of the training pipeline with respect to the performance on the
actual validation/test set. Without any augmentation, the classes that have a small number of labelled birds will not have 
much robustness. As in, if we change the image of a some bird (with only a few samples of this bird) ever-so-slightly, the 
model will perform poorly. This is why making some augmented data is important, as the model will learn much more sophisticated
features to identify and separate classes of birds.
{% highlight  linenos%}
img_transform = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=256),
        transforms.RandomRotation(45),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]),
    # The two subsets below must have the same transformations for our results to make sense.
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]),
}
{% endhighlight %}
<br>
<h3>Setting up the DataLoaders for the training loop</h3>
Setting up the DataLoaders for training loop in a typical PyTorch fashion. Since we have gigabytes of sample data,
I thought it was better to take the lion's share of the data into the training and not the validation or testing data.
I think that validation and testing data are simply for checking, since they do not directly benefit the learning of the 
model, only if I change hyperparameters because of the validation performance indicating a problem of some sort.
{% highlight  linenos%}
num_train_samples = len(glob.glob(DIR_PATH + 'train/*'))
indices = list(np.arange(num_train_samples))
random.shuffle(indices)
train_set_size = int(len(indices) * 0.95)
valid_set_size = len(indices) - train_set_size
train_indices, val_indices = torch.utils.data.random_split(indices, [train_set_size, valid_set_size])
dataset_train = torchvision.datasets.ImageFolder(root=PATH+'train', transform=img_transform['train'])
dataset_val = torchvision.datasets.ImageFolder(root=PATH+'train', transform=img_transform['val'])
test_set = torchvision.datasets.ImageFolder(root=PATH+'test', transform=img_transform['test'])

image_datasets = {'train': dataset_train, 'val': dataset_val}
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True, num_workers=4) for x in ['train', 'val']}
{% endhighlight %}
<br>
<h3>Initialization of various models (we use ResNet)</h3>
{% highlight linenos %}
# Using Resnet-50 from HuggingFace pretrained
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

# Initialize the model for this run
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

# Print the model we just instantiated
print(model_ft)
{% endhighlight %}
<br>
<h3>Optimizer and Parameter Update Setup</h3>
The output of this cell shows that we are only updating the very top layer of the ResNet model and nothing else.
Initially, I was messing around with just using Stochastic Gradient Descent, but this proved 
to be very slow at getting above 50% accuracy (took around 7 hours and 15 epochs). Eventually, I caught on to the most 
widely used flavor of Adam (AdamW) and this achieved 50% validation accuracy in just one epoch!
{% highlight linenos %}
# Send the model to GPU
model_ft = model_ft.to(device)

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# Observe that all parameters are being optimized
#optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9) Sucks, only get to 50% after like 15 epochs compared to just 1 with AdamW
optimizer_ft = optim.AdamW(params_to_update, lr=0.001)
{% endhighlight %}
<br>
<h3>Finally, training the model and saving the predictions</h3>
{% highlight linenos %}
# Initialize training with PyTorch training loop
import time
import copy
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = model_ft.to(device)
# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

# Train and evaluate
model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))
{% endhighlight %}
{% highlight linenos %}
def predict(net, dataloader, ofname):
    out = open(ofname, 'w')
    out.write("path,class\n")
    net.to(device)
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader, 0):
            if i%100 == 0:
                print(i)
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            fname, _ = dataloader.dataset.samples[i]
            out.write("test/{},{}\n".format(fname.split('/')[-1], data['to_class'][predicted.item()]))
    out.close()
testloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2)
checkpoints = "checkpoints/"
predict(resnet, test_loader, checkpoints+'submissions.csv')
{% endhighlight %}
Here are the best working hyperparameters and their scores:
<ol>
  <li>ResNet-18, 1 Unfrozen layer, AdamW, 10 epochs
    <dl>
      <dd>- Validation Accuracy: %55</dd>
      <dd>- Test Accuracy: * did not submit into test set *</dd>
    </dl>
  </li>
  <li>ResNet-18, All Layers Unfrozen, SGD, 15 epochs
    <dl>
      <dd>- Validation Accuracy: %45</dd>
      <dd>- Test Accuracy: * did not submit into test set *</dd>
    </dl>
  </li>
  <li>ResNet-18, All Layers Unfrozen, AdamW, 15 epochs
    <dl>
      <dd>- Validation Accuracy: %75</dd>
      <dd>- Test Accuracy: %74</dd>
    </dl>
  </li>
</ol>
<h1>Discussion Questions</h1>
<dl>
  <dt>1. What problems did you encounter?</dt>
  <dd>- Running out of GPU time</dd>
    <dd>&emsp;&emsp;* I ran out of GPU time on both Google Colab and Kaggle, which forced me to submit only 3 epochs (resnet on cpu takes too long and will be booted offline) of training for the test set model.</dd>
    <dd>&emsp;&emsp;* Dealing with downloading the data, transporting it to Google Colab, dealing with Google Colab glitches, and then running out of GPU time really set me back and made this project challenging.</dd>
    <dd>&emsp;&emsp;* Eventually, I had even called a friend to borrow her phone number to verify my account and use more GPU time on Kaggle.</dd>
  <dt>2. Are there next steps you would take if you kept working on the project?</dt>
  <dd>- Yes, I would've actually bought a GPU or some cloud GPU time on Kaggle/Colab so that I could train the model for longer and actually explore more hyperparameter tuning.</dd>
  <dd>- It ended up being quite uneventful because of the lack of GPU/compute resources to train these large vision models.</dd>
  <dt>3. How does your approach differ from others? Was that beneficial?</dt>
  <dd>- My approach didn't use any of the code from the other notebooks or the notebook on the CSE 455 website, so I couldn't ask anyone for help if something went wrong (not anyone but the TAs at least)</dd>
  <dd>- Although this was inconvenient, actually having to debug these things all on my own taught me alot about how kaggles system worked along with how PyTorch worked. I don't think I would have the same understanding if I was able to just ask a simple question and get it answered right away.</dd>
  <dd>- My approach <em>did</em> use the same general approach as the other competitors though. I think this is common practice to use a large vision model that was pretrained on ImageNet or Cifar or similar and finetuning on a task such as this one.</dd>
</dl>
Here is the link to my video describing my code and approach: [video](https://www.loom.com/share/db990997cdf24c3499f8b3b13d060903)
