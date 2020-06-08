#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predicting house prices: a regression example

"""

from sklearn.utils import shuffle
import numpy as np
from keras.preprocessing import image
import os
from keras.datasets import cifar100
from keras import models,layers
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.preprocessing.image import ImageDataGenerator


#Veri Setini Sadece resimleri alip depolamak icin yukluyorum.
#(train_data, train_targets), (test_data, test_targets) =\cifar100.load_data()
  

"""
Dosyalarin Ayarlanmasi ve Doldurulmasi


Train Datadaki verilerin dosya icindeki Kontrolu kolay olsun diye
burada hepsine bir degisken tanimladim ve dosya isimlerine gore kontrol yaptim
Tum train dosyalarinda 500'er Adet resim var.

#Tum Train Datayi Dolasacak
sayi = train_data.shape[0]
baby=0
caterpillar=0
mushroom=0
pickup_truck=0
shark=0
turtle=0  

from PIL import Image
for i in  range(sayi):
    
    im = Image.fromarray(train_data[i])
    if train_targets[i]==2:
        im.save("train\\baby\{}.png".format(baby))
        baby+=1
        
    if train_targets[i]==18:
        im.save("train\\caterpillar\{}.png".format(caterpillar))
        caterpillar+=1
    if train_targets[i]==51:
        im.save("train\\mushroom\{}.png".format(mushroom))
        mushroom+=1
    if train_targets[i]==58:
        im.save("train\\pickup_truck\{}.png".format(pickup_truck))
        pickup_truck+=1
    if train_targets[i]==73:
        im.save("train\\shark\{}.png".format(shark))
        shark+=1
    if train_targets[i]==93:
        im.save("train\\turtle\{}.png".format(turtle))
        turtle+=1

Butun Islemler Bu Sefer Test Datasi icin yapiliyor.Buradaki degiskenler kontrol
icin sifirlaniyor ve her Tum test dosyalarinda 100'er adet resim var.
  
sayi = test_data.shape[0]
baby=0
caterpillar=0
mushroom=0
pickup_truck=0
shark=0
turtle=0

for i in  range(sayi):
    
    im = Image.fromarray(test_data[i])
    if test_targets[i]==2:
        im.save("test\\baby\{}.png".format(baby))
        baby+=1
    if test_targets[i]==18:
        im.save("test\\caterpillar\{}.png".format(caterpillar))
        caterpillar+=1
    if test_targets[i]==51:
        im.save("test\\mushroom\{}.png".format(mushroom))
        mushroom+=1
    if test_targets[i]==58:
        im.save("test\\pickup_truck\{}.png".format(pickup_truck))
        pickup_truck+=1
    if test_targets[i]==73:
        im.save("test\\shark\{}.png".format(shark))
        shark+=1
    if test_targets[i]==93:
        im.save("test\\turtle\{}.png".format(turtle))
        turtle+=1
  """
"""
Bilgisayardaki train klasorundeki her bir dizine girilip resimler toplanir
Toplanan resimlerin verisinden train_data Olusturuluyor.
"""
Dosyalar = [os.path.join("train",fname) for fname in os.listdir("train")]
Say=0
Index=0
nb_classes = 6
TrainTarget = []

"""

6 Sinifi kullanin denildigi icin ben bana dusen siniflarin cifar100'deki hedef
degerlerini kullanmak yerine burada kendim deger verdim.

"""

for dosya in Dosyalar:
    TrainResimler = [os.path.join(dosya,fname) for fname in os.listdir(dosya)]
    if Say==0:
        x = np.array([np.array(image.load_img(res,target_size=(32,32))) for res in TrainResimler])
        TrainData = x  
        if dosya == "train\\baby":
            for Index in range(500):
                TrainTarget.append([1])
                
        if dosya == "train\\caterpillar":
            for Index in range(500):
                TrainTarget.append([2])
        
        if dosya == "train\\shark":
            for Index in range(500):
                TrainTarget.append([3])
        
        if dosya == "train\\mushroom":
            for Index in range(500):
                TrainTarget.append([4])
        
        if dosya == "train\\pickup_truck":
            for Index in range(500):
                TrainTarget.append([5])
        
        if dosya == "train\\turtle":
            for Index in range(500):
                TrainTarget.append([0])

        Say+=1
    else:
        
        if dosya == "train\\baby":
            for Index in range(500):
                TrainTarget.append([1])
                
        if dosya == "train\\caterpillar":
            for Index in range(500):
                TrainTarget.append([2])
        
        if dosya == "train\\shark":
            for Index in range(500):
                TrainTarget.append([3])
        
        if dosya == "train\\mushroom":
            for Index in range(500):
                TrainTarget.append([4])
        
        if dosya == "train\\pickup_truck":
            for Index in range(500):
                TrainTarget.append([5])
        
        if dosya == "train\\turtle":
            for Index in range(500):
                TrainTarget.append([0])
                
        x = np.array([np.array(image.load_img(res,target_size=(32,32))) for res in TrainResimler])
        TrainData = np.vstack((TrainData, x))

#Ayni sekilde test verilerininde hepsi cekiliyor.

Dosyalar = [os.path.join("test",fname) for fname in os.listdir("test")]
Say=0
Index=0

TestTarget = []

for dosya in Dosyalar:
    TrainResimler = [os.path.join(dosya,fname) for fname in os.listdir(dosya)]
    if Say==0:
        x = np.array([np.array(image.load_img(res,target_size=(32,32))) for res in TrainResimler])
        TestData = x  
        if dosya == "test\\baby":
            for Index in range(100):
                TestTarget.append([1])
                
        if dosya == "test\\caterpillar":
            for Index in range(100):
                TestTarget.append([2])
        
        if dosya == "test\\shark":
            for Index in range(100):
                TestTarget.append([3])
        
        if dosya == "test\\mushroom":
            for Index in range(100):
                TestTarget.append([4])
        
        if dosya == "test\\pickup_truck":
            for Index in range(100):
                TestTarget.append([5])
        
        if dosya == "test\\turtle":
            for Index in range(100):
                TestTarget.append([0])
        Say+=1
    else:
        
        if dosya == "test\\baby":
            for Index in range(100):
                TestTarget.append([1])
                
        if dosya == "test\\caterpillar":
            for Index in range(100):
                TestTarget.append([2])
        
        if dosya == "test\\shark":
            for Index in range(100):
                TestTarget.append([3])
        
        if dosya == "test\\mushroom":
            for Index in range(100):
                TestTarget.append([4])
        
        if dosya == "test\\pickup_truck":
            for Index in range(100):
                TestTarget.append([5])
        
        if dosya == "test\\turtle":
            for Index in range(100):
                TestTarget.append([0])
                
        x = np.array([np.array(image.load_img(res,target_size=(32,32))) for res in TrainResimler])
        TestData = np.vstack((TestData, x))


#Ag Modeli Kuruluyor. 3 Konv 2 Dense Katmani Var
model=models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation="relu",input_shape=(32,32,3)))

model.add(layers.Conv2D(64,(2,2),activation="relu"))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64,(2,2),activation="relu"))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(128,(2,2),activation="relu"))

model.add(layers.Flatten())
#model.add(layers.Dropout(0.5))
model.add(layers.Dense(64,activation="relu"))
#Karar katmani oldugundan Sinif Sayisi parametre olarak veriliyor.
model.add(layers.Dense(nb_classes,activation="softmax"))
model.summary()


#Optimizasyon
model.compile(optimizer="adam",
               loss='categorical_crossentropy',
               metrics=['accuracy'])

"""
#Ag Zenginlestirme Yeri

#Test ve Train konumlari ayarlaniyor.
train_dir = 'train'
test_dir = 'test'
train_datagen = ImageDataGenerator(zoom_range=0.3,#0.3 Oraninda Yaklastirma
                                   rotation_range=25,#25 Derece Dondurme Uygulaniyor
                                   horizontal_flip=True)#Yatay Ters Cevirme
test_datagen = ImageDataGenerator(zoom_range=0.3,
                                   rotation_range=25,
                                   horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=10,
                                                    target_size=(32,32),
                                                    class_mode="categorical",
                                                    shuffle=True)

test_generator = test_datagen.flow_from_directory(test_dir,
                                                        target_size=(32,32),
                                                        batch_size=10,
                                                        class_mode="categorical",
                                                        shuffle=True)
#Ag Egitimi Yapiliyor.
h = model.fit_generator(train_generator,
                              steps_per_epoch=300,
                              epochs=15,
                              validation_data=test_generator,
                              validation_steps=50)

#Veri Zenginlestirme ile egitilen ag test ediliyor.
model.evaluate_generator(generator=test_generator, steps=20)
#Ag Zenginlestirme Yeri

"""
TrainTargetArray = np.array(TrainTarget) 
TestTargetArray = np.array(TestTarget)
#Categorical Crossentropy Kullanabilmek icin Hedef Verileri Categorilendiriliyor.
TrainTargets = np_utils.to_categorical(TrainTargetArray, nb_classes)
TestTargets = np_utils.to_categorical(TestTargetArray, nb_classes)


TrainTargets,TrainData = shuffle(TrainTargets,TrainData, random_state=0)
TestData,TestTargets = shuffle(TestData,TestTargets,random_state=0)

#Egitim
h=model.fit(TrainData, 
           TrainTargets,
           epochs=15,
           batch_size=4,
           verbose=1,
           validation_data=(TestData, TestTargets)
           )
"""
#Gorsel Grafik Islemleri
epochs=range(1,1+len(h.history['loss']))
plt.plot(epochs,h.history['val_loss'])
plt.plot(epochs,h.history['loss'])
plt.plot(epochs,h.history['val_accuracy'])
plt.plot(epochs,h.history['accuracy'])
plt.xlabel('epoch')
plt.ylabel('loss')
"""
"""

epochs=range(1,1+len(h.history['loss']))
plt.plot(epochs,h.history['loss'])
plt.xlabel('Epoch Sayisi')
plt.ylabel('Loss')

epochs=range(1,1+len(h.history['val_loss']))
plt.plot(epochs,h.history['val_loss'])
plt.xlabel('Epoch Sayisi')
plt.ylabel('Validation Loss')


"""