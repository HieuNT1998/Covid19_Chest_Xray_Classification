from csv import reader
import cv2, random
import matplotlib.pyplot as plt
import os 
import numpy as np
import torch

covid_path = './data/covid19'
normal_path = './data/normal'


witdh = 224
height = 224


def read_img(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img,(height,witdh))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = torch.tensor(img)
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
    num_train = (length*2*8)//10
    return images[:num_train],labels[:num_train],images[(num_train+1):],labels[(num_train+1):]


def load_path():
    covid_image_path = []
    normal_image_path = []
    
    # get normal
    image_names = os.listdir(normal_path)
    for i in image_names:
        normal_image_path.append(os.path.join(normal_path,i))
    
    # get covid_image_path 
    image_names = []
    with open('metadata.csv', 'r') as read_obj:
        csv_reader = reader(read_obj)
        for row in csv_reader:
            if row[4]=='COVID-19' and row[16] == 'PA' : 
                image_names.append(row[21])

    for i in image_names:
        covid_image_path.append(os.path.join(normal_path,i))
    
    length = len(normal_image_path)

    ## init label
    covid_label = [1 for i in range(length)]
    normal_label = [0 for i in range(length)]

    image_paths = covid_image_path + normal_image_path
    labels = covid_label + normal_label

    z = list(zip(image_paths,labels))
    random.shuffle(z)
    image_paths,labels = zip(*z)

    print(image_paths)
    return image_paths,labels



def load_batch(list_path):
    list_img = []
    
    for i in list_path:
        img = read_img()
        list_img.append(img)

    list_img = torch.Tensor(list_img)
    return list_img


if __name__ == '__main__':
    load_data()