#importing all the required libraries

import re
import nltk # to classify text
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lm = WordNetLemmatizer()    #lemmatizer

from nltk.tokenize import TreebankWordTokenizer
tk = TreebankWordTokenizer()    #tokenizer

import pickle   # to load tfidf vectorizer used in text classification.

import cv2      # to manage images
import pandas as pd
from keras.models import load_model     # to load models

def images_predict(image_file_names):
    model = load_model('components/Saved_models/images_model.h5')    # load model from saved models.

    predictions = []

    # images = glob.glob(image_dir+"/*.jpg")      # load paths of all images into images variable ( Note: All images should be in .jpg extension ).
    for image_name in image_file_names:       # iterate through all the images.

        image_path = 'components/recieved_images/'+image_name
        print(image_path)
        img = cv2.imread(image_path)        # read image from image path.
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)       # convert the image using cv2.
        img = cv2.resize(img,(256,256))     # resize the image to (256,256) , because it is the image size used while training the model.
        img = img.reshape(1,256,256,-1)     # convert the image tnto numpy array.
        img = img/255

        pred = model.predict(img)       # predict the image. Pred containes 3 values which are probabilities of image being in 3 categories.

        max = 0     # to find max of all probabilities
        ind = 0     # to find the index of that max category.

        for i in range(len(pred[0])):
            if pred[0][i] > max:
                max = pred[0][i]
                ind = i
        if ind == 0:
            predictions.append('farmer related')
        elif ind == 1:
            predictions.append('offensive')
            return -1
        else:
            predictions.append('not farmer related')
            return 0

    return 1


def text_predict(text):
    text_model = load_model('components/Saved_models/text_model.h5')     # load text classification model from saved models.

    tfidf = pickle.load(open("components/Text_Classification_files/tfidf.pickel", "rb"))     # load the vectorizer from saved text classification files.

    df = pd.read_csv('components/Text_Classification_files/badwordsenglish.csv')     # read the csv file of badwords and convert it into list.
    bad_words2 = df['2g1c'].tolist()

    k = nltk.tokenize.sent_tokenize(text)
    flag = 0
    correct = 0
    cor = []
    total = 0
    for x in k:
        if flag == 0:
            x = re.sub('[^a-zA-Z]',' ',x)
            x = x.lower()
            flag = 0
            if flag == 0:
                x = tk.tokenize(x)
                x = [lm.lemmatize(word) for word in x if not word in stopwords.words('english')]
                for y in bad_words2:
                    if y in x:
                        flag = 1
                        break
                if flag == 0: 
                    x = ' '.join(x)
                    x = tfidf.transform([x]).toarray()
                    pred = text_model.predict(x)
                    cor.append(pred[0][0])
                    if pred[0][0] > 0.5:
                        correct += 1
                        total += 1
                    else:
                        total += 1
                else:
                    break
    if flag == 1:
        return -1
    elif correct/total > 0.4:
        return 1
    else:
        return 0

def predict(image_file_names,text):
    
    isImagesRelated = images_predict(image_file_names)
    if isImagesRelated == 1:
        isTextRelated = text_predict(text)
        if isTextRelated == 1:
            print("Blog is related.")
            return 1
        elif isTextRelated == 0:
            print("Blog text is not related")
            return 0
        else:
            print("Blog text is offensive")
            return -1
    elif isImagesRelated == 0:
        print("Blog images are not related")
        return 0
    else:
        print("Blog images are offencive")
        return -1