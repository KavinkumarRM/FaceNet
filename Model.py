import keras
import keras.backend as K
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout,Activation,Flatten,Input,concatenate,Lambda
from keras.optimizers import Adam
import numpy as np
from keras import regularizers
from keras.layers import Conv2D, MaxPooling2D,LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import model_from_json
from keras.losses import categorical_crossentropy
from sklearn.ensemble import GradientBoostingClassifier
import cv2
import os
from os import walk
from PIL import Image
from sklearn.model_selection import train_test_split
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib

filedir="./new_data/"
listofnames=[i for i in os.listdir(filedir) if not i.startswith(".") ]
num_classes=len(listofnames)

def img_to_matrix(imagePath):
    image=cv2.imread(imagePath)
    image=cv2.resize(image, (150,150))
    return image

X,Y=[],[]
for j in range(len(listofnames)):
    for i in os.listdir(filedir+listofnames[j]):
        if i.endswith(".jpeg"):
            X.append(img_to_matrix(filedir+listofnames[j]+"/"+i))
            Y.append(j)

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def data_generator(X):
    data=[]
    data.append(adjust_gamma(X,0.5))
    data.append(adjust_gamma(X,0.75))
    data.append(X)
    data.append(adjust_gamma(X,1.25))
    data.append(adjust_gamma(X,1.5))
    return(data)

def brightness_dict_func(X,Y):
    brightness_dict={}
    brightness_list=[]
    for i in range(len(Y)):
        if Y[i] not in brightness_dict.keys():
            brightness_dict[Y[i]]=[np.sum(X[i])/(np.prod(X[i].shape)*255)]
        else:
            brightness_dict[Y[i]].append(np.sum(X[i])/(np.prod(X[i].shape)*255))
        brightness_list.append(np.sum(X[i])/(np.prod(X[i].shape)*255))
    return(brightness_dict,brightness_list)

def data_augmentation(X,Y):
    new_X=[]
    new_Y=[]
    for i in range(len(Y)):
        new_X=new_X+data_generator(X[i])
        new_Y=new_Y+[Y[i]]*5
    return(new_X,new_Y)

X,Y=data_augmentation(X,Y)

def preprocess(X,Y):
    flat_X = np.array(X)
    flat_Y = np.array(Y)
    flat_X = flat_X.astype('float32')
    flat_X/=255
    return flat_X,flat_Y

X,Y=preprocess(X,Y)
X = X.reshape(X.shape[0], 150, 150, 3)

X_preserved,Y_preserved=X,Y

def resnet(X,Y):
    inputs = Input(shape=X)
    conv1 = Conv2D(32,(3,3), strides = (1,1), padding='same',kernel_initializer="lecun_uniform",
                       kernel_regularizer=regularizers.l2(0))(inputs)
    batch1 = BatchNormalization()(conv1)
    activation1 = Activation('relu')(batch1)
    max1 = MaxPooling2D(pool_size=(2, 2))(activation1)
    drop1 = Dropout(0.25)(max1)

    conv2 = Conv2D(128,(5,5), strides = (1,1), padding='same',kernel_regularizer=regularizers.l2(0))(drop1)
    batch2 = BatchNormalization()(conv2)
    activation2 = Activation('relu')(batch2)
    max2 = MaxPooling2D(pool_size=(2, 2))(activation2)
    drop2 = Dropout(0.25)(max2)

    conv3 = Conv2D(512,(3,3), strides = (1,1), padding='same',kernel_regularizer=regularizers.l2(0))(drop2)
    batch3 = BatchNormalization()(conv3)
    activation3 = Activation('relu')(batch3)
    max3 = MaxPooling2D(pool_size=(2, 2))(activation3)
    drop3 = Dropout(0.25)(max3)

    conv4 = Conv2D(512,(3,3), strides = (1,1), padding='same',kernel_regularizer=regularizers.l2(0))(drop3)
    batch4 = BatchNormalization()(conv4)
    activation4 = Activation('relu')(batch4)
    max4 = MaxPooling2D(pool_size=(2, 2))(activation4)
    drop4 = Dropout(0.1)(max4)

    flat = Flatten()(drop4)

    dense1 = Dense(256,kernel_initializer="lecun_uniform")(flat)
    activation5 = Activation('tanh')(dense1)
    drop5 = Dropout(0.25)(activation5)
    dense2 = Dense(Y)(drop5)
    output = Activation('softmax')(dense2)
    model = Model(input=inputs,output=output,name='residual_network')
    return(model)

def facenet2(X):
    inputs = Input(shape=X)
    conv11 = Conv2D(32,(3,3), strides = (1,1), padding='same',kernel_initializer="lecun_uniform",
                       kernel_regularizer=regularizers.l2(0))(inputs)
    batch11 = BatchNormalization()(conv11)
    activation11 = Activation('relu')(batch11)
    max11 = MaxPooling2D(pool_size=(2, 2))(activation11)
    drop11 = Dropout(0.25)(max11)

    conv21 = Conv2D(128,(5,5), strides = (1,1), padding='same',kernel_regularizer=regularizers.l2(0))(drop11)
    batch21 = BatchNormalization()(conv21)
    activation21 = Activation('relu')(batch21)
    max21 = MaxPooling2D(pool_size=(2, 2))(activation21)
    drop21 = Dropout(0.25)(max21)

    conv31 = Conv2D(512,(3,3), strides = (1,1), padding='same',kernel_regularizer=regularizers.l2(0))(drop21)
    batch31 = BatchNormalization()(conv31)
    activation31 = Activation('relu')(batch31)
    max31 = MaxPooling2D(pool_size=(2, 2))(activation31)
    drop31 = Dropout(0.25)(max31)

    conv41 = Conv2D(512,(3,3), strides = (1,1), padding='same',kernel_regularizer=regularizers.l2(0))(drop31)
    batch41 = BatchNormalization()(conv41)
    activation41 = Activation('relu')(batch41)
    max41 = MaxPooling2D(pool_size=(2, 2))(activation41)
    drop41 = Dropout(0.1)(max41)

    conv111 = Conv2D(512,(3,3), strides = (4,4), padding='same',kernel_initializer="lecun_uniform",
                       kernel_regularizer=regularizers.l2(0))(inputs)
    batch111 = BatchNormalization()(conv111)
    activation111 = Activation('relu')(batch111)
    max111 = MaxPooling2D(pool_size=(4, 4))(activation111)
    drop111 = Dropout(0.25)(max111)

    drop4 = concatenate([drop41,drop111])

    conv411 = Conv2D(512,(3,3), strides = (1,1), padding='same',kernel_regularizer=regularizers.l2(0))(drop4)
    batch411 = BatchNormalization()(conv411)
    activation411 = Activation('relu')(batch411)
    max411 = MaxPooling2D(pool_size=(2, 2))(activation411)
    drop411 = Dropout(0.1)(max411)



    flat2 = Flatten()(drop411)

    dense1 = Dense(256,kernel_initializer="lecun_uniform")(flat2)
    activation5 = Activation('tanh')(dense1)
    drop5 = Dropout(0.25)(activation5)
    dense2 = Dense(64)(drop5)
    output = Activation('sigmoid')(dense2)
    model = Model(input=inputs,output=output)
    return(model)

def facenet(X):
    inputs = Input(shape=X)
    conv1 = Conv2D(32,(3,3), strides = (1,1), padding='same',kernel_initializer="lecun_uniform",
                       kernel_regularizer=regularizers.l2(0))(inputs)
    batch1 = BatchNormalization()(conv1)
    activation1 = Activation('relu')(batch1)
    max1 = MaxPooling2D(pool_size=(2, 2))(activation1)
    drop1 = Dropout(0.25)(max1)

    conv2 = Conv2D(128,(5,5), strides = (1,1), padding='same',kernel_regularizer=regularizers.l2(0))(drop1)
    batch2 = BatchNormalization()(conv2)
    activation2 = Activation('relu')(batch2)
    max2 = MaxPooling2D(pool_size=(2, 2))(activation2)
    drop2 = Dropout(0.25)(max2)

    conv3 = Conv2D(512,(3,3), strides = (1,1), padding='same',kernel_regularizer=regularizers.l2(0))(drop2)
    batch3 = BatchNormalization()(conv3)
    activation3 = Activation('relu')(batch3)
    max3 = MaxPooling2D(pool_size=(2, 2))(activation3)
    drop3 = Dropout(0.25)(max3)

    conv4 = Conv2D(512,(3,3), strides = (1,1), padding='same',kernel_regularizer=regularizers.l2(0))(drop3)
    batch4 = BatchNormalization()(conv4)
    activation4 = Activation('relu')(batch4)
    max4 = MaxPooling2D(pool_size=(2, 2))(activation4)
    drop4 = Dropout(0.1)(max4)

    flat = Flatten()(drop4)

    dense1 = Dense(256,kernel_initializer="lecun_uniform")(flat)
    activation5 = Activation('tanh')(dense1)
    drop5 = Dropout(0.25)(activation5)
    dense2 = Dense(12)(drop5)
    output = Activation('sigmoid')(dense2)
    model = Model(input=inputs,output=output)
    return(model)

def triplet_loss(x):
    anchor, positive, negative = tf.split(x, 3,axis=-1)

    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)

    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), 0.9)
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)

    return loss

def build_model(input_shape):

    K.set_image_data_format('channels_last')
    p = Input(shape=input_shape)
    n = Input(shape=input_shape)
    a = Input(shape=input_shape)
    emb = facenet2(input_shape)
    p_emb,n_emb,a_emb = emb(p),emb(n),emb(a)
    merged_output = concatenate([a_emb, p_emb, n_emb])
    triplet_los = Lambda(triplet_loss, output_shape=(1,),name = 'triplet_los')(merged_output)

    #residual = Input(shape=input_shape)
    #res = resnet(input_shape,num_class)
    #residual_output = res(p)
    #residual_loss = Lambda(categorical_crossentropy(Y1), output_shape=(1,))(residual_output)

    #merged_final_output = concatenate([triplet_los,residual_loss])

    model = Model(inputs=[a, p, n],outputs=triplet_los)
    #model.compile(loss='mean_absolute_error', optimizer=Adam(learning_rate=0.0001))

    return model


def dict_ensemble(X,i):
    rest=[]
    for j in X.keys():
        if j!=i:
            rest=rest+list(X[j])
    return(rest)

def get_all_triplet(X,Y):
    unique_y=list(set(Y))
    X_dict={}
    index_dict={}
    for i in unique_y:
        X_dict[i]=X[[j for j in range(len(Y)) if Y[j]==i]]
        index_dict[i]=[j for j in range(len(Y)) if Y[j]==i]
    a,p,n=[],[],[]
    index_a,index_p,index_n=[],[],[]
    for i in unique_y:
        values=X_dict[i]
        indexs=index_dict[i]
        counter_values=dict_ensemble(X_dict,i)
        counter_index=dict_ensemble(index_dict,i)
        for j in range(len(values)):
            for k in range(len(values)):
                if j!=k:
                    for l in range(len(counter_values)):
                            a.append(values[j])
                            p.append(values[k])
                            n.append(counter_values[l])
                            index_a.append(indexs[j])
                            index_p.append(indexs[k])
                            index_n.append(counter_index[l])
    return (a,p,n,index_a,index_p,index_n)

data_a,data_p,data_n,i_a,i_p,i_n=get_all_triplet(X,Y)
master_index=range(len(data_a))

model=build_model(data_a[0].shape)
model.compile(optimizer=Adam(learning_rate=0.00001), loss='mean_absolute_error')


def stratify_sample(pred,num_triplet):
    dist=distance_matrix(pred,pred)
    dist_dict={}
    for i in master_index:
        dist_dict[i]=dist[i_a[i]][i_p[i]]-dist[i_a[i]][i_n[i]]
    sorted_list=[k for k, v in sorted(dist_dict.items(), key=lambda item: item[1] ,reverse=True)]
    return(sorted_list[:num_triplet])

def train_model(model,number_of_runs,num_triplet,epochs,initial_sample):
    sample_triplet= lambda x: random.sample(master_index,x)
    list_man = lambda lists,k:[lists[i] for i in k]
    model_list,loss_list=[],[]
    if not initial_sample:
        initial_sample=sample_triplet(num_triplet)

        a,p,n=list_man(data_a,initial_sample),list_man(data_p,initial_sample),list_man(data_n,initial_sample)
    else:
        print('\tcomputing triple distance \n')
        innermodel=model.layers[3]
        intra_prediction=innermodel.predict(X)
        print('\tresampling \n')
        stratified_sample=stratify_sample(intra_prediction,num_triplet)
        a,p,n=list_man(data_a,stratified_sample),list_man(data_p,stratified_sample),list_man(data_n,stratified_sample)

    for i in range(number_of_runs):
        print('run:',i,'\n')

        callback = keras.callbacks.EarlyStopping(monitor='loss', patience=max(1,epochs-1))
        model.fit(x=[a,p,n],y=np.zeros(len(a)), batch_size=1,epochs=epochs,callbacks=[callback])
        print('\n\tsaving intermidiate model\n')
        exec("model.save('./model_history_offline/intermidiate_model_run{0}.h5')".format(i))

        if i!=number_of_runs-1:
            print('\tcomputing triple distance \n')
            innermodel=model.layers[3]
            intra_prediction=innermodel.predict(X)

            print('\tresampling \n')
            stratified_sample=stratify_sample(intra_prediction,num_triplet)
            a,p,n=list_man(data_a,stratified_sample),list_man(data_p,stratified_sample),list_man(data_n,stratified_sample)

    return(model)

model=train_model(model,5,3000,2,True)

innermodel=model.layers[3]
innermodel.save('./facenet_resampling_1.h5')

model_json = innermodel.to_json()
with open("model_resampling_1.json", "w") as json_file:
    json_file.write(model_json)

pred=innermodel.predict(X)

GBM=GradientBoostingClassifier()
GBM.fit(pred,Y)
predicted_class=GBM.predict(pred)
sum(predicted_class==Y)/len(Y)

RF_classifier=RandomForestClassifier()
RF_classifier.fit(pred,Y)
predicted_class=RF_classifier.predict(pred)
sum(predicted_class==Y)/len(Y)

classifier=SVC()
classifier.fit(pred,Y)
predicted_class=classifier.predict(pred)
sum(predicted_class==Y)/len(Y)

_ = joblib.dump(GBM, './GBM_resampling_1.joblib.pkl', compress=9)
_ = joblib.dump(RF_classifier, './RF_resampling_1.joblib.pkl', compress=9)
_ = joblib.dump(classifier, './SVM_resampling_1.joblib.pkl', compress=9)
