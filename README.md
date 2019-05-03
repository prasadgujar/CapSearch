#    **CapSearch**
 An Automated Image Caption Generation and Caption Based Image Retrieval Application.

## contents

* [Introduction](https://github.com/prasadgujar/CapSearch/new/master?readme=1#introduction)
* [Features](https://github.com/prasadgujar/CapSearch/new/master?readme=1#features)
* [Demo Video](https://github.com/prasadgujar/CapSearch/new/master?readme=1#video)
* [Dataset and Pre-trained models](https://github.com/prasadgujar/CapSearch/new/master?readme=1#dataset-and-pre-trained-models)
* [Packages Required](https://github.com/prasadgujar/CapSearch/new/master?readme=1#packages-required)
* [Future Scope](https://github.com/prasadgujar/CapSearch/new/master?readme=1#what-you-can-expect-in-future-versions)
* [Publication](https://github.com/prasadgujar/CapSearch/new/master?readme=1#paper-publication)
* [Contributors](https://github.com/prasadgujar/CapSearch/new/master?readme=1#Contributors)
* [Contribute](https://github.com/prasadgujar/CapSearch/new/master?readme=1#Contribute)
* [Acknowledgement](https://github.com/prasadgujar/CapSearch/new/master?readme=1#Acknowledgement)

## Introduction
This is a python (Flask Application) based Automated Image Caption and Image Retrieval model which makes use of deep learning image caption generator. It uses a merge model comprising of Convolutional Neural Network (CNN) and a Long Short Term Memory Network (LSTM) . The dataset used here is Flickr8K dataset.

## Features

This model can be used via GUI. In model-

* Automated Caption Generation (Offline) - Upload Image and retrive automated caption based on image features.
* Caption Based Image Search (Similar Images) - Given Text Based Query and it will return similar images based on image caption and similarity.

## Video
[![CapSearch](https://i.ytimg.com/vi/W9pRkQAp3sc/maxresdefault.jpg)](https://www.google.com/url?sa=i&source=images&cd=&cad=rja&uact=8&ved=2ahUKEwiLq_jLvf_hAhWBs48KHYESDzIQjRx6BAgBEAU&url=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DW9pRkQAp3sc&psig=AOvVaw2FSdxfhVrWVSmztDeWl6TG&ust=1556977032981102)

## Dataset and Pre-trained models
* [Flicker-8k-Dataset](https://forms.illinois.edu/sec/1713398)

## Packages Required:
* Anaconda
* Keras with Tensorflow Backend (Python 3.6)
* Flask

## What you can expect in future versions?
* Make a highly scalable REST API which accepts the image and returns the caption of the image
* Make a dashboard through which the training of the captioner could be done on custom datasets.
* Introduce unit tests and logging to enable smooth debugging.
* Improve the caption based image search part for the more accuracy.
* Make a dashboard through which user can manage their image database.
* Improve the UI part of the application.
* Change the architecture of image captioner in order reduce the memory footprint required by the current pre trained models
* Further development may also include working on improvising with more accurate predictions and search results

## Paper Publication:
*  [CapSearch - “ An Image Caption Generation based search”](https://www.irjet.net/archives/V6/i4/IRJET-V6I4760.pdf)

## Contributor
 * [Prasad Gujar](https://github.com/prasadgujar)
 * [Shaunak Baradkar](https://github.com/shaunVB)
 * [Aditya Bhatia](https://github.com/audi187)

## Contribute
* Fork this repository and contribute.
* Feel free to report bugs.
* All types of feedbacks are welcome

## Acknowledgement

* A special thanks to [Machine Learning Mastery](https://machinelearningmastery.com/) without which we couldn't have thought about the right approach to tackle this problem.
