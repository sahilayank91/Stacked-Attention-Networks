A Project for the Visual Question and Answering using the Stacked attention networks.

To run 
1) Download the MS COCO dataset from the COCO website .http://cocodataset.org/
2) Store int the data/image folder for the respective train and test datasets.
3) Download the COCO QA from here https://www.cs.toronto.edu/~mren/imageqa/data/cocoqa/
4) Copy the Image ids from the test and train and then store in the data folder in the respective train, test and their question and answer folder.
5) Run utils/get_data.py

It will generate the Image, Question and answers feature in the pickle files.

Use the Pickle files to generate and train the different models like MLP(Multi layer perceptron),2 layer SAN and 3 layer SAN

