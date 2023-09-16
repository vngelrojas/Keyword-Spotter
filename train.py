import keras

DATA_PATH = "data.json"
SAVED_MODEL_PATH = "model.h5"
LEARNING_RATE = 0.0001
EPOCHS = 40 # number of times data is passed through the network for training
BATCH_SIZE = 32

def loadDataset(data_path):
    pass

def getDataSplits(data_path):
    # load dataset
     X,y = loadDataset(data_path)
    # create train/validation/test splits
    
    # convert inputs from 2d to 3d arrays
    pass
def main():
    # load train/validation/test data splits
    X_train,X_validation,X_test,y_train,y_validation,y_test = getDataSplits(DATA_PATH)
    
    # build the CNN model
    input_data = (X_train.shape[1],X_train.shape[2],X_train.shape[3]) # (num of segemnts,num of coeff (13), num of depth channels (1))
    model = buildModel(input_shape, LEARNING_RATE)
    
    # train the model
    model.fit(X_train,y_train,epochs=EPOCHS, batch_size = BATCH_SIZE, validation_data=(X_validation,y_validation))
    
    #evaluate the model
    test_error,test_accuracy = model.evaluate(X_test,y_test)
    print(f"Test error: {test_error}, Test accuracy: {test_accuracy}")
    
    #save the model
    model.save(SAVED_MODEL_PATH)
    