from feat import Detector
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt

#################################################################
################### Helper Functions ############################
#################################################################

## process batch detection once (like batch from deprecated detector function)
## latest version doesn't have it, 
## see the issue and my comment: https://github.com/cosanlab/py-feat/issues/238#issuecomment-2481365979
def processImages(img_list, detector):
    for img in img_list:
        detection = detector.detect_image(img)  
        ## yield for lazy loading (since we don't know the input size)
        yield img, detection 

## return the dominant emotion from a detection row
def getDominantEmotion(detection): 
    return detection.emotions.idxmax()

## create csv file from processed detections (output_csv)
def createCSV(processed_detections, output_csv):
    file_exists = os.path.exists(output_csv)
    for img, detection in processed_detections:
        ## here copying only the aus because .loc operation had some problems while 
        ## modifying the detection.aus view. Instead I copy it as a freash df object
        ## since I only load&yield once, I might need other fields later on
        detection_aus = detection.aus.copy()
        ## append new file/face cols to the dataframe
        detection_aus['file'] = os.path.basename(img)
        ## detection has ([0] [0 -> Row] [1 -> Row]) |  ([1] -> [0 -> Row] [1 -> Row]) type of structure 
        detection_aus['face'] = range(len(detection_aus))

        ## get all cols, add file and face to the start of it, then write it to detection again
        cols = ['file', 'face'] + [col for col in detection_aus.columns if col not in ['file', 'face']]
        detection_aus = detection_aus[cols]
        
        ## write if it doesn't exist already, otherwise keep appending without writing new headers
        detection_aus.to_csv(output_csv, mode='a' if file_exists else 'w', header=not file_exists, index=False)
        file_exists = True  

## annotate the face boxes on the images and output to output_dir
def visualizeImages(processed_detections, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for img, detection in processed_detections:
        ## load the img with cv2
        image = cv2.imread(img)

        ## I iterate over dataframes here for multiple face AUs
        for _, row in detection.iterrows():
            ## get the starting coordinates: 
            x, y = int(row['FaceRectX']), int(row['FaceRectY'])

            ## get the w/h
            width, height = int(row['FaceRectWidth']), int(row['FaceRectHeight'])

            ## calculate and draw the rect
            start_point = (int(x), int(y))
            end_point = (int(x+width), int(y+height))
            cv2.rectangle(image, start_point, end_point, (0, 0, 255), 3)

            ## add calculated max emotion label from sfp fex
            cv2.putText(image, getDominantEmotion(row), (int(x), int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
        
        ## place the annotated image to ./output dir
        output_path = os.path.join(output_dir, os.path.basename(img))
        cv2.imwrite(output_path, image)

def loadValences(filePath):
    valences_read = pd.read_csv(filePath)
    ## load csv to valences_dict
    valences_dict = dict(zip(valences_read['file'], valences_read['valence']))
    return valences_dict

## create two dicts with positive and negative au dataframes
def groupAUs(processed_detections, valences_dict):
    positive_AUs = []
    negative_AUs = []

    for img, detection in processed_detections:
        ## casting for getting AUs (like in the docs)
        type(detection)
        img_name_no_extension = os.path.basename(img).split('.')[0] 
        valence_val = valences_dict.get(img_name_no_extension)

        if valence_val == 'positive':
            positive_AUs.append(detection.aus)
        elif valence_val == 'negative':
            negative_AUs.append(detection.aus)

    return positive_AUs, negative_AUs

def calculateMean(AU_group):
    concated_dataframe = pd.concat(AU_group)
    all_AU_means = concated_dataframe.mean(axis=0)
    return all_AU_means

def plot_sorted_aus(sorted_differences, output_file):
    au_labels = sorted_differences.index
    differences = sorted_differences.values

    plt.figure(figsize=(10, 10))
    plt.plot(au_labels, differences, marker='o', linestyle='', markersize=8, label='abs dif')
    plt.xlabel('AUs', fontsize=12)
    plt.ylabel('abs mean diff. ', fontsize=12)
    plt.title('Absolute difference in AU means(pos. vs. neg.)', fontsize=14)

    plt.tight_layout()
    plt.savefig(output_file)

#################################################################
#################################################################
#################################################################

detector = Detector()
test_dir_path = os.path.join(os.getcwd(), 'dataset/images/')
img_list = [os.path.join(test_dir_path, img) for img in os.listdir(test_dir_path)]

## process images once and reuse detections/since latest py-feat version doesn't allow batch call
processed_detections = list(processImages(img_list, detector))

## Step 1: create the output imgs with annotations
visualizeImages(processed_detections, 'processed/images/')

## Step 2: create the csv with file name/face nums attached
createCSV(processed_detections, 'processed/aus.csv')

## Step 3: sort AUs according to positive and negative annotations | calculate mean for all AU types
positiveAUs, negativeAUs = groupAUs(processed_detections, loadValences('dataset/annotations.csv')) 

positive_means = calculateMean(positiveAUs)
negative_means = calculateMean(negativeAUs)

absolute_differences = abs(positive_means - negative_means)
sorted_differences = absolute_differences.sort_values(ascending=False)

## Step 4: Plot abs mean dif. vs AU group graph
plot_sorted_aus(sorted_differences, 'processed/au_visualization.png')


