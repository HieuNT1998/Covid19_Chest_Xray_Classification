from csv import reader
import cv2, random
import matplotlib.pyplot as plt
import os 
import numpy as np
import torch

path = '../covid-chestxray-dataset/images/'

witdh = 800
height = 800

def load_covid_image():
    image_names = []
    with open('metadata.csv', 'r') as read_obj:
        csv_reader = reader(read_obj)
        for row in csv_reader:
            if row[4]=='COVID-19' and row[16] == 'PA' : image_names.append(row[21])
    
    image_paths = []
    for i in image_names:
        image_paths.append(path + i)
    
    covid_list = []
    for i in image_paths:
        img = cv2.imread(i)
        img = cv2.resize(img,(height,witdh))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        covid_list.append(img)
    return covid_list
    
def load_normal_image():
    image_paths = os.listdir('./data/normal')
    normal_list = []
    for i in image_paths:
        path = './data/normal/' + i
        img = cv2.imread(path)
        img = cv2.resize(img,(height,witdh))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        normal_list.append(img)
    return normal_list

def load_data():
    ### load data
    covid_list = load_covid_image()
    normal_list = load_normal_image()
    length = len(normal_list)
    
    ## init lable
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
