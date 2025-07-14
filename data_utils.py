import numpy as np
import os
import scipy.io as sio
from config import *


EXPERIMENT_LABELS_IV = [[1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
                        [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1],
                        [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]]

EXPERIMENT_LABELS_V = [[4, 1, 3, 2, 0, 4, 1, 3, 2, 0, 4, 1, 3, 2, 0],
[2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0],
[2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0]]


def loadSEED(path, modality):
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    label_mat = sio.loadmat(label_mat_unx)
    label_set = label_mat['label']
    for i in range(1,16):
        if modality == 'EEG':
            filename = 'eeg_de_{}.npy'.format(i)
        elif modality == "Eye":
            filename = 'eye_{}.npy'.format(i)
        npyFile = os.path.join(path,filename) # .......npy
        file_detail = np.load(npyFile) #sample_num * 310
        n_sample, dim = file_detail.shape
        if i < 10:
            train_data.extend(file_detail)
            train_label.extend(np.repeat((label_set[0][i-1]+1),n_sample))
        else:
            test_data.extend(file_detail)
            test_label.extend(np.repeat((label_set[0][i-1]+1),n_sample))
    print('Loaded training shape data: {}, label: {}'.format(str(np.shape(train_data)), str(np.shape(train_label))))
    print('Loaded test shape data: {}, label: {} '.format(str(np.shape(test_data)), str(np.shape(test_label))))
    return np.array(train_data), np.array(train_label), np.array(test_data), np.array(test_label)

def loadSEEDIV(path,si, modality):
    si=int(si)-1
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    for i in range(1,25): # there was a bug here iterating from 1 to 24. But it should not influence the result significantly. 
        if modality == 'EEG':
            filename = 'eeg_de_{}.npy'.format(i)
        elif modality == "Eye":
            filename = 'eye_{}.npy'.format(i)
        npyFile = os.path.join(path,filename) # .......npy
        file_detail = np.load(npyFile) #sample_num * 310
        file_detail = np.nan_to_num(file_detail)
        assert (True in np.isnan(file_detail)) == False
        n_sample, dim = file_detail.shape
        if i < 17:
            train_data.extend(file_detail)
            train_label.extend(np.repeat((EXPERIMENT_LABELS_IV[si][i-1]),n_sample))
        else:
            test_data.extend(file_detail)
            test_label.extend(np.repeat((EXPERIMENT_LABELS_IV[si][i-1]),n_sample))
    print('Loaded training shape data: {}, label: {}'.format(str(np.shape(train_data)), str(np.shape(train_label))))
    print('Loaded test shape data: {}, label: {} '.format(str(np.shape(test_data)), str(np.shape(test_label))))
    return np.array(train_data), np.array(train_label), np.array(test_data), np.array(test_label)

def loadSEEDV_ThreeFold(path,fold, modality):
    # There are three sessions for each subject. Every session contains 15 movie clips. 
    # Put first five clips from each session together as one fold. In the end, every subject has one result.
    all_index = [[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]]
    test_index = all_index[fold]
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    # fold number is the test set
    for session in range(3):
        if modality == 'EEG':
            session_path = os.path.join(path, str(session))
        elif modality == "Eye":
            session_path = os.path.join(path, str(int(session)+1))
        

        for i in range(1,16):
            if modality == 'EEG':
                filename = 'DE_{}.npy'.format(i-1)
            elif modality == "Eye":
                filename = 'eye_{}.npy'.format(i)
            npyFile = os.path.join(session_path,filename) # .......npy
            file_detail = np.load(npyFile) #sample_num * 310

            if modality == 'EEG':
                dim,freq,n_sample = file_detail.shape
                file_detail = file_detail.reshape(310,n_sample)
                file_detail = np.transpose(file_detail)
                n,d = file_detail.shape
            elif modality == "Eye":
                d_eye, n_eye= file_detail.shape   
                file_detail = np.transpose(file_detail) 
                n_sample,d = file_detail.shape
            # 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
            # 
            if i in test_index:
                test_data.extend(file_detail)
                test_label.extend(np.repeat((EXPERIMENT_LABELS_V[session][i-1]),n_sample))
            else:
                train_data.extend(file_detail)
                train_label.extend(np.repeat((EXPERIMENT_LABELS_V[session][i-1]),n_sample))
    print('Loaded training shape data: {}, label: {}'.format(str(np.shape(train_data)), str(np.shape(train_label))))
    print('Loaded test shape data: {}, label: {} '.format(str(np.shape(test_data)), str(np.shape(test_label))))
    return np.array(train_data), np.array(train_label), np.array(test_data), np.array(test_label)


def overlapDataForTimeWindow(X_test, X_train, time_window):
    #overlap窗口
    dataDim = X_test.shape[-1] # 310 for eeg, 23 for eye
    n_sample_train = np.shape(X_train)[0]
    n_sample_test = np.shape(X_test)[0]
    eeg_data_train=np.zeros((0, dataDim))

    for time_number in range(n_sample_train-time_window):
        eeg_data_train = np.row_stack((eeg_data_train, X_train[time_number:time_number+time_window]))
    for time_number in range(time_window):
        eeg_data_train = np.row_stack((eeg_data_train, X_train[n_sample_train-time_window:n_sample_train]))

    eeg_data_test=np.zeros((0, dataDim))

    for time_number in range(n_sample_test-time_window):
        eeg_data_test = np.row_stack((eeg_data_test, X_test[time_number:time_number+time_window]))
    for time_number in range(time_window):
        eeg_data_test = np.row_stack((eeg_data_test, X_test[n_sample_test-time_window:n_sample_test]))

    eeg_data_train = eeg_data_train.reshape(-1, time_window, dataDim)
    eeg_data_test = eeg_data_test.reshape(-1, time_window,dataDim)
    assert n_sample_train == np.shape(eeg_data_train)[0] 
    assert n_sample_test == np.shape(eeg_data_test)[0]

    return eeg_data_test, eeg_data_train


def splitTrainTest(dataset, person_folder,session,fold):
    if dataset == "seed":
        temp_eeg_path = os.path.join(unx_path_eeg, person_folder)
        eeg_path = os.path.join(temp_eeg_path, session)
        train_data_eeg, train_label_eeg, test_data_eeg, test_label_eeg = loadSEED(eeg_path,'EEG')

        temp_eye_path = os.path.join(unx_path_eye, person_folder)
        eye_path = os.path.join(temp_eye_path, session)
        train_data_eye, _, test_data_eye, _ = loadSEED(eye_path,'Eye')
    elif dataset == "seediv":
        temp_eeg_path = os.path.join(unx_path_eeg_iv, person_folder)
        eeg_path = os.path.join(temp_eeg_path, session)
        train_data_eeg, train_label_eeg, test_data_eeg, test_label_eeg = loadSEEDIV(eeg_path,session,'EEG')
        temp_eye_path = os.path.join(unx_path_eye_iv, person_folder)
        eye_path = os.path.join(temp_eye_path, session)
        train_data_eye, _, test_data_eye, _ = loadSEEDIV(eye_path,session,'Eye')
    else:
        temp_eeg_path = os.path.join(unx_path_eeg_v, person_folder)
        train_data_eeg, train_label_eeg, test_data_eeg, test_label_eeg = loadSEEDV_ThreeFold(temp_eeg_path,fold,'EEG')
        temp_eye_path = os.path.join(unx_path_eye_v, person_folder)
        train_data_eye, _, test_data_eye, _ = loadSEEDV_ThreeFold(temp_eye_path,fold,'Eye')

    return train_data_eeg, train_label_eeg, test_data_eeg, test_label_eeg,train_data_eye, test_data_eye
