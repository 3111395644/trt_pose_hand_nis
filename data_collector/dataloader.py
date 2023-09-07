import os
import json
import cv2
import numpy as np

class dataloader:
    
    
    def __init__(self, path, train_label, test_label):
        self.train_path = path+"/hand_dataset_train_nis/train/"
        self.test_path = path+"/hand_dataset_test_nis/test/"
        self.train_label_path = path+train_label
        self.test_label_path = path+test_label
        
        self.train_data = []
        self.train_images  =[]
        self.train_file_name = []
        
        self.test_data = []
        self.test_images = []
        self.test_file_name = []
        
        self.labels_train = []
        self.labels_test = []
        
        self.load_hand_dataset(self.train_path, self.test_path)
        self.load_labels(self.train_label_path, self.test_label_path)
        
        
        
       
        
    def _assert_exist(self, label_path):
        msg = 'File is not availble: %s' % label_path
        assert os.path.exists(label_path), msg
    def load_labels(self, train_label_path, test_label_path):
        """
        loads the training and testing labels from a json file
        """
        self._assert_exist(train_label_path)
        self._assert_exist(test_label_path)
        
        with open(train_label_path, 'r') as f:
            train_label_data = json.load(f)
        self.labels_train = train_label_data["labels"]
        
        with open(test_label_path, 'r') as f:
            test_label_data = json.load(f)
        self.labels_test = test_label_data["labels"]
        
        #return labels_train, labels_test
    def scaled_data(self, train_data, test_data):
        """
        This method helps scaling/normalizing data
        """
        raw_scaler = preprocessing.StandardScaler().fit(train_data)
        scaled_train_data = raw_scaler.transform(train_data)
        scaled_test_data = raw_scaler.transform(test_data)
        return scaled_train_data, scaled_test_data, raw_scaler
    def load_hand_dataset(self, train_path, test_path):
        """
        This method loads the images for training the classifier. 
        """
        WIDTH = 224
        HEIGHT = 224
        for filename in sorted(os.listdir(train_path)):
            self.train_file_name.append(filename)
            image = cv2.imread(train_path+filename)
            #image = image[:, ::-1, :]
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if image is None or image.size == 0:
                print("Input image is empty or size is zero.")
            else:
                print("Input image exists.")
            image = cv2.resize(image, (WIDTH, HEIGHT), interpolation = cv2.INTER_AREA)
            self.train_data.append(np.reshape(np.array(image), 3*WIDTH*HEIGHT))
            self.train_images.append(image)
     
        for filename in sorted(os.listdir(test_path)):
            self.test_file_name.append(filename)
            image = cv2.imread(test_path+filename)
            #image = image[:, ::-1, :]
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if image is None or image.size == 0:
                print("Input image is empty or size is zero.")
            else:
                print("Input image exists.")
            image = cv2.resize(image, (WIDTH, HEIGHT), interpolation = cv2.INTER_AREA)
            self.test_images.append(image)
            self.test_data.append(np.reshape(np.array(image), 3*WIDTH*HEIGHT))
        #return train_images, test_images, train_data, test_data
    def smaller_dataset(self, dataset, no_samples_per_class, no_of_classes):
        """
        This method helps training with a subset of the dataset 
        """
        total_samples_per_class =100
        start = 0
        end = no_samples_per_class
        new_dataset = []
        labels = []
        for i in range(no_of_classes):
            new_data = dataset[start:end]
            start = start+total_samples_per_class
            end = start+no_samples_per_class
            new_dataset.extend(new_data)
            labels.extend([i+1]*no_samples_per_class)
        return new_dataset, labels           
