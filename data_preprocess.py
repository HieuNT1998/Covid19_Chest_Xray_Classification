from csv import reader
import cv2, random
import matplotlib.pyplot as plt
import os 
import numpy as np
import torch

covid_path = './data/covid19'
normal_path = './data/normal'


witdh = 800
height = 800


def read_img(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img,(height,witdh))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img    


def load_covid_image():
    image_names = []
    with open('metadata.csv', 'r') as read_obj:
        csv_reader = reader(read_obj)
        for row in csv_reader:
            if row[4]=='COVID-19' and row[16] == 'PA' : 
                image_names.append(row[21])

    covid_list = []
    for i in image_names:
        image_path = os.path.join(covid_path,i)
        covid_list.append(read_img(image_path))
    return covid_list
    

def load_normal_image():
    image_names = os.listdir(normal_path)
    normal_list = []
    for i in image_names:
        image_path = os.path.join(normal_path,i)
        normal_list.append(read_img(image_path))
    return normal_list


def load_data():
    ### load data
    covid_list = load_covid_image()
    normal_list = load_normal_image()
    length = len(normal_list)

    ## init label
    covid_label = [1 for i in range(length)]
    normal_label = [0 for i in range(length)]

    ## shuffle data
    images = covid_list + normal_list
    labels = covid_label + normal_label
    z = list(zip(images,labels))
    random.shuffle(z)
    images,labels = zip(*z)
    
    ## split train and val
    num_train = int(((length*2)/10) * 8)
    images = np.array(images)
    labels = np.array(labels)
    return images[:num_train,:,:,:],labels[:num_train],images[(num_train+1):,:,:,:],labels[(num_train+1):]


x_train,y_train,x_valid,y_valid = load_data()
print("label", y_train[0])
plt.imshow(x_train[0])
plt.show()
