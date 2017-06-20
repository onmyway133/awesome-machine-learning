# fantastic-machine-learning
I like to explore machine learning, but don't want the to dive into other platforms, like Python or Javascript, to understand some frameworks, or TensorFlow. Luckily, at WWDC 2017, Apple introduces Core ML, Vision, ARKit, which makes working with machine learning so much easier. With all the pre-trained models, we can build great things. It's good to feel the outcome first, then try to explore advanced topics and underlying mechanisms 🤖

This will curates things mostly related to Core ML, and Swift. There are related things in other platforms if you want to get some references

I just learn so I don't want to include very advanced, low level stuff in this list 😇

## Table of contents

- [Core ML](#core-ml)
- [Vision](#vision)
- [Natural Language Processing](#natural-language-processing)
- [Metal](#metal)
- [GamePlayKit](#gameplaykit)
- [Artificial Intelligence](#artificial-intelligence)
- [Speech Recognition](#speech-recognition)
- [General Learning](#general-learning)
- [Misc](#misc)

## Core ML

### General

- [Machine Learning](https://developer.apple.com/machine-learning/) Build more intelligent apps with machine learning.

### Introduction

- [Introducing Core ML](https://developer.apple.com/videos/play/wwdc2017/703/)
- [Core ML in depth](https://developer.apple.com/videos/play/wwdc2017/710/)
- [Core ML and Vision: Machine Learning in iOS 11 Tutorial](https://www.raywenderlich.com/164213/coreml-and-vision-machine-learning-in-ios-11-tutorial)
- [iOS 11: Machine Learning for everyone](http://machinethink.net/blog/ios-11-machine-learning-for-everyone/)
- [Everything a Swift Dev Ever Wanted to Know About Machine Learning](https://news.realm.io/news/swift-developer-on-machine-learning-try-swift-2017-gallagher/)

### Models :rocket:

- [caffe](https://github.com/BVLC/caffe) Caffe: a fast open framework for deep learning. http://caffe.berkeleyvision.org/
- [deep-learning-models](https://github.com/fchollet/deep-learning-models) Keras code and weights files for popular deep learning models.
- [tensorflow models](https://github.com/tensorflow/models) Models built with TensorFlow
- [libSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) A Library for Support Vector Machines
- [scikit-learn](http://scikit-learn.org/) Machine Learning in Python
- [xgboost](https://github.com/dmlc/xgboost) Scalable, Portable and Distributed Gradient Boosting (GBDT, GBRT or GBM) Library, for Python, R, Java, Scala, C++ and more. Runs on single machine, Hadoop, Spark, Flink and DataFlow
- [Keras-Classification-Models](https://github.com/titu1994/Keras-Classification-Models) Collection of Keras models used for classification
- [MobileNet-Caffe](https://github.com/shicai/MobileNet-Caffe) Caffe Implementation of Google's MobileNets

### Tools

- [coremltools](https://pypi.python.org/pypi/coremltools) coremltools in a python package for creating, examining, and testing models in the .mlmodel format

### Tutorials

- [Swift Tutorial: Native Machine Learning and Machine Vision in iOS 11](https://hackernoon.com/swift-tutorial-native-machine-learning-and-machine-vision-in-ios-11-11e1e88aa397)

### Examples

- [Core-ML-Sample](https://github.com/yulingtianxia/Core-ML-Sample) A Demo using Core ML Framework
- [UnsplashExplorer-CoreML](https://github.com/ahmetws/UnsplashExplorer-CoreML) Core ML demo app with Unsplash API
- [MNIST_DRAW](https://github.com/hwchong/MNIST_DRAW) This is a sample project demonstrating the use of Keras (Tensorflow) for the training of a MNIST model for handwriting recognition using CoreML on iOS 11 for inference.

## Vision

### General

- [Vision](https://developer.apple.com/documentation/vision) Apply high-performance image analysis and computer vision techniques to identify faces, detect features, and classify scenes in images and video.

### Guide

- [Blog-Getting-Started-with-Vision](https://github.com/jeffreybergier/Blog-Getting-Started-with-Vision)
- [Swift World: What’s new in iOS 11 — Vision](https://medium.com/compileswift/swift-world-whats-new-in-ios-11-vision-456ba4156bad)

## Natural Language Processing

### General

- [NSLinguisticTagger](https://developer.apple.com/documentation/foundation/nslinguistictagger) Analyze natural language to tag part of speech and lexical class, identify proper names, perform lemmatization, and determine the language and script (orthography) of text.

### Guide

- [Linguistic Tagging](https://www.objc.io/issues/7-foundation/linguistic-tagging/)
- [NSLinguisticTagger on NSHipster](http://nshipster.com/nslinguistictagger/)

### Repos

- [CoreLinguistics](https://github.com/rxwei/CoreLinguistics) This repository contains some fundamental data structures for NLP.
- [SwiftVerbalExpressions](https://github.com/VerbalExpressions/SwiftVerbalExpressions) Swift Port of VerbalExpressions

## Metal

### General

- [Metal](https://developer.apple.com/metal/)

### Guide

- [MPSCNNHelloWorld: Simple Digit Detection Convolution Neural Networks (CNN)](https://developer.apple.com/library/content/samplecode/MPSCNNHelloWorld/Introduction/Intro.html)
- [MetalImageRecognition: Performing Image Recognition](https://developer.apple.com/library/content/samplecode/MetalImageRecognition/Introduction/Intro.html)
- [Apple’s deep learning frameworks: BNNS vs. Metal CNN](http://machinethink.net/blog/apple-deep-learning-bnns-versus-metal-cnn/)

### Repos

- [Forge](https://github.com/hollance/Forge) A neural network toolkit for Metal

## GamePlayKit

### General

- [GamePlayKit](https://developer.apple.com/documentation/gameplaykit)

### Guide

- [Project 34: Four in a Row](https://www.hackingwithswift.com/read/34/overview)
- [GKMinmaxStrategist: What does it take to build a TicTacToe AI?](http://tilemapkit.com/2015/07/gkminmaxstrategist-build-tictactoe-ai/)
- [GameplayKit Tutorial: Artificial Intelligence](https://www.raywenderlich.com/146407/gameplaykit-tutorial-artificial-intelligence)

## Artificial Intelligence

### Posts

- [The classic ELIZA chat bot in Swift.](https://gist.github.com/hollance/be70d0d7952066cb3160d36f33e5636f)
- [Introduction to AI Programming for Games](https://www.raywenderlich.com/24824/introduction-to-ai-programming-for-games)

## Speech Recognition

### General

- [Speech](https://developer.apple.com/documentation/speech)

### Guide

- [Using the Speech Recognition API in iOS 10](https://code.tutsplus.com/tutorials/using-the-speech-recognition-api-in-ios-10--cms-28032)
- [Speech Recognition Tutorial for iOS](https://www.raywenderlich.com/155752/speech-recognition-tutorial-ios)

### Repos

- [CeedVocal](https://github.com/creaceed/CeedVocal) Speech Recognition Library for iOS

## General Learning

### Overview

- [A visual introduction to machine learning](http://www.r2d3.us/visual-intro-to-machine-learning-part-1/)
- [Machine Learning is Fun!](https://medium.com/@ageitgey/machine-learning-is-fun-80ea3ec3c471)
- [10 Machine Learning Terms Explained in Simple English](http://blog.aylien.com/10-machine-learning-terms-explained-in-simple/)

### How to learn

- [Machine Learning in a Year](https://medium.com/learning-new-stuff/machine-learning-in-a-year-cdb0b0ebd29c)
- [Machine Learning Self-study Resources](https://ragle.sanukcode.net/articles/machine-learning-self-study-resources/)
- [How to Learn Machine Learning](https://elitedatascience.com/learn-machine-learning)
- [Getting Started with Machine Learning](https://medium.com/@suffiyanz/getting-started-with-machine-learning-f15df1c283ea)
- [The Non-Technical Guide to Machine Learning & Artificial Intelligence](https://machinelearnings.co/a-humans-guide-to-machine-learning-e179f43b67a0)

### Guide

- [Machine Learning: An In-Depth Guide - Overview, Goals, Learning Types, and Algorithms](http://www.innoarchitech.com/machine-learning-an-in-depth-non-technical-guide/)
- [A Tour of Machine Learning Algorithms](http://machinelearningmastery.com/a-tour-of-machine-learning-algorithms/)
- [Machine Learning for Hackers](https://www.youtube.com/playlist?list=PL2-dafEMk2A4ut2pyv0fSIXqOzXtBGkLj)
- [Machine Learning for Developers For absolute beginners and fifth graders](https://xyclade.github.io/MachineLearning/)
- [dive-into-machine-learning](https://github.com/hangtwenty/dive-into-machine-learning) Dive into Machine Learning with Python Jupyter notebook and scikit-learn
- [An introduction to machine learning with scikit-learn](http://scikit-learn.org/stable/tutorial/basic/tutorial.html)
- [Building powerful image classification models using very little data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)

### Guide in Swift

- [Machine Learning for iOS](https://www.invasivecode.com/weblog/machine-learning-swift-ios/)
- [Use TensorFlow and BNNS to Add Machine Learning to your Mac or iOS App](https://www.bignerdranch.com/blog/use-tensorflow-and-bnns-to-add-machine-learning-to-your-mac-or-ios-app/)
- [The “hello world” of neural networks](http://machinethink.net/blog/the-hello-world-of-neural-networks/)
- [Convolutional Neural Networks in iOS 10 and macOS](https://www.invasivecode.com/weblog/convolutional-neural-networks-ios-10-macos-sierra/)
- [LearningMachineLearning](https://github.com/graceavery/LearningMachineLearning) Swift implementation of "Data Science From Scratch" and http://karpathy.github.io/neuralnets/
- [Getting started with TensorFlow on iOS](http://machinethink.net/blog/tensorflow-on-ios/)

### Courses

- [6.S191: Introduction to Deep Learning](http://introtodeeplearning.com/index.html)
- [Machine Learning](http://introtodeeplearning.com/index.html)

### Interview

- [41 Essential Machine Learning Interview Questions](https://www.springboard.com/blog/machine-learning-interview-questions/)

## Misc

### Other ML frameworks

- [TensorSwift](https://github.com/qoncept/TensorSwift) A lightweight library to calculate tensors in Swift, which has similar APIs to TensorFlow's
- [Swift-AI](https://github.com/Swift-AI/Swift-AI) The Swift machine learning library.
- [Swift-Brain](https://github.com/vlall/Swift-Brain) Artificial intelligence/machine learning data structures and Swift algorithms for future iOS development. bayes theorem, neural networks, and more AI.
- [EmojiIntelligence](https://github.com/Luubra/EmojiIntelligence) Neural Network built in Apple Playground using Swift
- [Bender](https://github.com/xmartlabs/Bender) Easily craft fast Neural Networks on iOS! Use TensorFlow models. Metal under the hood.
- [BrainCore](https://github.com/aleph7/BrainCore) The iOS and OS X neural network framework
- [AIToolbox](https://github.com/KevinCoble/AIToolbox) A toolbox of AI modules written in Swift: Graphs/Trees, Support Vector Machines, Neural Networks, PCA, K-Means, Genetic Algorithms
- [brain](https://github.com/harthur/brain) Neural networks in JavaScript
- [TensorFlow](https://www.tensorflow.org/get_started/mnist/beginners) An open-source software library for Machine Intelligence
- [incubator-predictionio](https://github.com/apache/incubator-predictionio) PredictionIO, a machine learning server for developers and ML engineers. Built on Apache Spark, HBase and Spray.
- [Caffe](http://caffe.berkeleyvision.org/) Deep learning framework by BAIR
- [Torch](http://torch.ch/) A SCIENTIFIC COMPUTING FRAMEWORK FOR LUAJIT
- [Theano](http://www.deeplearning.net/software/theano/) Theano is a Python library that allows you to define, optimize, and evaluate mathematical expressions involving multi-dimensional arrays efficiently
- [CNTK](https://github.com/Microsoft/CNTK) Microsoft Cognitive Toolkit (CNTK), an open source deep-learning toolkit
- [MXNet](http://mxnet.io/) Lightweight, Portable, Flexible Distributed/Mobile Deep Learning

### Accelerate

- [Accelerate-in-Swift](https://github.com/hyperjeff/Accelerate-in-Swift) Swift example codes for the Accelerate.framework
- [Surge](https://github.com/mattt/Surge) A Swift library that uses the Accelerate framework to provide high-performance functions for matrix math, digital signal processing, and image manipulation.

### Statistics

- [SigmaSwiftStatistics](https://github.com/evgenyneu/SigmaSwiftStatistics) A collection of functions for statistical calculation written in Swift

### Linear Algebra

- []()

### Services

- [Watson](https://www.ibm.com/watson/developercloud/) Enable cognitive computing features in your app using IBM Watson's Language, Vision, Speech and Data APIs.
- [wit.ai](https://wit.ai/) Natural Language for Developers
- [Cloud Machine Learning Engine](https://cloud.google.com/ml-engine/) Machine Learning on any data, any size
- [Cloud Vision API](https://cloud.google.com/vision/) Derive insight from images with our powerful Cloud Vision API
- [Amazon Machine Learning](https://aws.amazon.com/documentation/machine-learning/) Amazon Machine Learning makes it easy for developers to build smart applications, including applications for fraud detection, demand forecasting, targeted marketing, and click prediction
- [api.ai](https://api.ai/) Build brand-unique, natural language interactions for bots, applications, services, and devices.
- [clarifai](https://developer.clarifai.com/) Build amazing apps with the world’s best image and video recognition API.
- [openml](https://www.openml.org/) Exploring machine learning together
