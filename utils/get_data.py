import os
import pickle
from get_image_features import get_image_features
from get_question_features import get_question_features


image_train_features = [0]*1000
question_train_features = [0]*1000
image_test_features = [0]*1000
questions_test_features = [0]*1000
answers_train = []
answers_test = []
image_ids_train = []
image_ids_test = []
id_list_train = []
id_list_test = []
index_id_to_remove_train = []
index_id_to_remove_test = []


"""
To get a list of training image and question features
"""
img_id_path = "/home/iiitk/SAN/data/images/train/img_ids.txt"
fp = open(img_id_path)
for id in fp:
    id = id.strip("\n")
    id_list_train.append(id)
fp.close()


def get_training_image_data():
    i = 0
    path_images = "/home/iiitk/SAN/data/images/train/train2014/" # Define correct path
    for filename in os.listdir(path_images):
        if filename == "img_ids.txt":
            continue
        filename = filename[15:]
        filename = filename.rstrip(".jpg")
        id = filename.strip("0")
        if id in id_list_train:
            i = i + 1
            filename = path_images + "COCO_train2014_" + filename + ".jpg"
            img_feature = get_image_features(filename)
            index = id_list_train.index(id)
            print(str(i) + ") " + str(index))
            image_ids_train.append(index)
            image_train_features[index] = img_feature

    with open('image_features_train.pkl', 'wb') as f:
        pickle.dump(image_train_features, f)



"""
def get_correct_question_id_train():
    id_to_remove = [x for x in id_list_train if x not in image_ids_train]
    for id in id_to_remove:
        index_id_to_remove_train.append(id_list_train.index(id))
    return index_id_to_remove_train
"""


def get_training_question_data():
    fp = open("/home/iiitk/SAN/data/questions/train/question.txt") #Provide correct path
    for index, question in enumerate(fp):
        question = question.strip("\n")
        question_feature = get_question_features(question)
        question_train_features.append(question_feature)
    with open('question_features_train.pkl', 'wb') as f:
        pickle.dump(question_train_features, f)



def get_training_answers():
    fp = open("/home/iiitk/SAN/data/answers/train/answer.txt")
    list_train = []
    for ans in fp:
        ans = ans.strip("\n")
        if ans not in list_train:
            list_train.append(ans)
        else:
            continue
    fp.close()
    list_train = sorted(list_train)
    fp = open("/home/iiitk/SAN/data/answers/train/answer.txt")
    for index, answer in enumerate(fp):
        answer = answer.strip("\n")
        answer = list_train.index(answer)
        answers_train.append(answer)

    with open('answers_train.pkl', 'wb') as f:
        pickle.dump(answers_train, f)





"""
To get a list of test image and question features
"""

img_id_path = "/home/iiitk/SAN/data/images/test/img_ids.txt"
fp = open(img_id_path)
for id in fp:
    id = id.strip("\n")
    id_list_test.append(id)
fp.close()


def get_test_image_data():
    i = 0
    path_images = "/home/iiitk/SAN/data/images/test/test2014/" # Define correct path
    for filename in os.listdir(path_images):
        if filename == "img_ids.txt":
            continue
        i = i+1
        filename = filename[15:]
        filename = filename.rstrip(".jpg")
        id = filename.strip("0")
        if id in id_list_test:
            filename = "COCO_train2014_" + filename + ".jpg"
            img_feature = get_image_features(filename)
            index = id_list_test.index()
            image_ids_test.append(index)
            image_test_features[index] = img_feature

    with open('image_features_test.pkl', 'wb') as f:
        pickle.dump(image_test_features, f)


"""
def get_correct_question_id_test():
    id_to_remove = [x for x in id_list_test if x not in image_ids_test]
    for id in id_to_remove:
        index_id_to_remove_test.append(id_list_test.index(id))
    return index_id_to_remove_test
"""


def get_test_question_data():
    fp = open("/home/iiitk/SAN/data/questions/test/question.txt") # Provide correct path
    for index, question in enumerate(fp):
        question = question.strip("\n")
        question_feature = get_question_features(question)
        questions_test_features.append(question_feature)

    with open('question_features_test.pkl', 'wb') as f:
        pickle.dump(questions_test_features, f)


def get_test_answers():
    fp = open("/home/iiitk/SAN/data/answers/test/answer.txt")
    list_test = []
    for ans in fp:
        ans = ans.strip("\n")
        if ans not in list_test:
            list_test.append(ans)
        else:
            continue
    fp.close()

    fp = open("/home/iiitk/SAN/data/answers/test/answer.txt")
    for index, answer in enumerate(fp):
        answer = answer.strip("\n")
        answer = list_test.index(answer)
        answers_test.append(answer)

    with open('answers_test.pkl', 'wb') as f:
        pickle.dump(answers_test, f)


get_training_image_data()
# get_training_question_data()
# get_training_answers()
# get_test_image_data()
# get_test_question_data()
# get_test_answers()

