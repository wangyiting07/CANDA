import torch.utils.data as Data
from torch.autograd import Variable
from data_utils import *
from model import *
import os
import torch
import torch.nn as nn
import numpy as np
import pickle
import itertools
import logging
import datetime
import copy

from parsing import parse_train_args
args = parse_train_args()





def model(train_eeg, test_eeg, train_eye, test_eye,train_label, test_label,
  alpha, common_dim, wd,LR, common_dim_tranf, depth, heads):
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 32

    train_eeg = torch.from_numpy(train_eeg).to(torch.float).to(device)
    train_eye = torch.from_numpy(train_eye).to(torch.float).to(device)
    test_eeg = torch.from_numpy(test_eeg).to(torch.float).to(device)
    test_eye = torch.from_numpy(test_eye).to(torch.float).to(device)
    train_label = torch.from_numpy(train_label).to(torch.long).to(device)
    test_label = torch.from_numpy(test_label).to(torch.long).to(device)
        
    train_eeg_dataset = Data.TensorDataset(train_eeg, train_label)   #499*5*310
    train_eeg_loader = Data.DataLoader(
        dataset=train_eeg_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False)

    train_eye_dataset = Data.TensorDataset(train_eye, train_label)    #499*5*41
    train_eye_loader = Data.DataLoader(
        dataset=train_eye_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False)

    test_eeg_dataset = Data.TensorDataset(test_eeg, test_label)     #343*5*310
    test_eeg_loader = Data.DataLoader(
        dataset=test_eeg_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False) 

    test_eye_dataset = Data.TensorDataset(test_eye, test_label)
    test_eye_loader = Data.DataLoader(
        dataset=test_eye_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False)

    encoder_eeg = Encoder_Transformer(EEG_DIM,depth, heads, int(EEG_DIM/2), common_dim_tranf).to(device)
    encoder_eye = Encoder_Transformer(EYE_DIM,depth, heads, int(EYE_DIM/2), common_dim_tranf).to(device)
    decoder_eeg = Decoder_Transformer(EEG_DIM,depth, heads, int(EEG_DIM/2), common_dim_tranf).to(device)
    decoder_eye = Decoder_Transformer(EYE_DIM,depth, heads, int(EYE_DIM/2), common_dim_tranf).to(device)
    cross_encoder = Cross_Attention_Layer_2(EEG_DIM,EYE_DIM,depth, heads, int(EEG_DIM/2), common_dim_tranf).to(device)
    emotion_classifier = Classifier_Emotion(common_dim, OUTPUT_DIM).to(device)
    discriminator = Discriminator(common_dim).to(device)



    criterion = nn.CrossEntropyLoss().to(device)
    adversarial_loss = torch.nn.BCELoss().to(device)
    pixelwise_loss = torch.nn.L1Loss(size_average=False).to(device) # loss between decoder and original data

    optimizer_generator = torch.optim.Adam(
            itertools.chain(encoder_eeg.parameters(), encoder_eye.parameters(), decoder_eeg.parameters(),
                            decoder_eye.parameters(), emotion_classifier.parameters(),cross_encoder.parameters()), 
                             weight_decay=wd, lr=LR)
    
    optimizer_discriminator = torch.optim.Adam(
            itertools.chain(discriminator.parameters()), weight_decay=wd, lr=LR)

    best_acc = 0.0
    for epoch in range(1,args.epochs):
        logging.critical("epoch: {}".format(epoch))
        avg_train_loss = 0.0
        avg_rloss = 0.0
        avg_closs = 0.0
        avg_test_loss = 0.0
        avg_gloss = 0.0
        avg_dloss = 0.0
        eeg_dataloader_iteratior = iter(train_eeg_loader)
        correct_train_eeg = 0.0
        correct_train_eye = 0.0
        for i, (data_eye, label_train) in enumerate(train_eye_loader):
            try:
                data_eeg, _ = next(eeg_dataloader_iteratior)
            except StopIteration:
                eeg_dataloader_iteratior = iter(train_eeg_loader)
                data_eeg, _  = next(eeg_dataloader_iteratior)
            #encode and dcoder process
            valid = Variable(torch.cuda.FloatTensor(label_train.shape[0], 1).fill_(1.0), requires_grad=False).to(device)
            fake = Variable(torch.cuda.FloatTensor(label_train.shape[0], 1).fill_(0.0), requires_grad=False).to(device)
            
            optimizer_generator.zero_grad()
            optimizer_discriminator.zero_grad()

          
            x_eeg_en = encoder_eeg(data_eeg) 
            x_eye_en = encoder_eye(data_eye) 
           
            #cross attention with original input
            cross_en = cross_encoder(data_eeg, data_eye)
         
            if epoch % 3 != 0:
                optimizer_discriminator.zero_grad()
                x_eeg_dis = discriminator(x_eeg_en) 
                x_eye_dis = discriminator(x_eye_en) 
                x_cross_attention = discriminator(cross_en)
                real_loss = adversarial_loss(x_eye_dis, valid)
                
                fake_loss = alpha * adversarial_loss(x_eeg_dis, fake) + (1-alpha)*adversarial_loss(x_cross_attention,fake)
                d_loss = 0.5 * (real_loss + fake_loss)
                avg_dloss += d_loss.item()
                d_loss.backward(retain_graph=True)

                optimizer_discriminator.step()

            if epoch % 3 == 0:
                optimizer_generator.zero_grad()
                b,d = x_eye_en.shape
                recon_eeg_encoder_output = x_eeg_en.reshape(b, 6, -1 )
                recon_eye_encoder_output = x_eye_en.reshape(b, 6, -1 )
                x_eeg_recon = decoder_eeg(recon_eeg_encoder_output)
                x_eye_recon = decoder_eye(recon_eye_encoder_output)
                x_eeg_dis = discriminator(x_eeg_en)
                x_cross_attention = discriminator(cross_en)
                eeg_class = emotion_classifier(x_eeg_en)
                eye_class = emotion_classifier(x_eye_en)
                cross_class = emotion_classifier(cross_en)

                g_loss = alpha * adversarial_loss(x_eeg_dis,valid)+(1-alpha)*adversarial_loss(x_cross_attention,valid) #+(1-alpha) * (rl1)
                avg_gloss += g_loss.item()
                g_loss.backward(retain_graph=True)
                
                re_loss = pixelwise_loss(x_eeg_recon, data_eeg) + pixelwise_loss(x_eye_recon, data_eye)
                avg_rloss += re_loss.item()
                re_loss.backward(retain_graph=True)

                c_loss = criterion(eeg_class,label_train)+criterion(eye_class,label_train)+criterion(cross_class,label_train)
                avg_closs += c_loss.item()
                c_loss.backward(retain_graph=True)
                pred_train_eye = eye_class.data.max(1)[1]  # get the index of the max log-probability
                correct_train_eye += pred_train_eye.eq(label_train.data).cpu().sum()
                pred_train_eeg = eeg_class.data.max(1)[1]  # get the index of the max log-probability
                correct_train_eeg += pred_train_eeg.eq(label_train.data).cpu().sum()

                optimizer_generator.step() 
                
        
        acc_train_eye = correct_train_eye / len(train_eye_loader.dataset)
        acc_train_eeg = correct_train_eeg / len(train_eye_loader.dataset)
       
           
        
        avg_rloss = avg_rloss/len(train_eye_loader.dataset)
        logging.info("pixelwise_loss loss: {}".format(avg_rloss))
        avg_gloss = avg_gloss/len(train_eye_loader.dataset)
        logging.info("generator loss: {}".format(avg_gloss))
        avg_closs = avg_closs/len(train_eye_loader.dataset)
        logging.info("classifier loss: {}".format(avg_closs))
        avg_dloss = avg_dloss/len(train_eye_loader.dataset)
        logging.info("discriminator loss: {}".format(avg_dloss))
        
        

        #test phrase
        encoder_eye.eval()
        emotion_classifier.eval()
        test_eeg_dataloader_iteratior = iter(test_eeg_loader)
        predict_label_all = []
        correct=0
        test_loss=0
        for i, (data_test_eye, _) in enumerate(test_eye_loader):
            try:
                _, lable_test_eeg = next(test_eeg_dataloader_iteratior)
            except StopIteration:
                test_eeg_dataloader_iteratior = iter(test_eeg_loader)
                _, lable_test_eeg  = next(test_eeg_dataloader_iteratior)
            #eeg_test_encoder = encoder_eeg(data_test_eeg)
            x_eye_en = encoder_eye(data_test_eye)
            eye_class = emotion_classifier(x_eye_en)

            loss_test = criterion(eye_class, lable_test_eeg)
            avg_test_loss += loss_test.item() / data_test_eye.shape[0]

            pred = eye_class.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(lable_test_eeg.data).cpu().sum()
            predict_label_all = np.append(predict_label_all, pred.data.cpu())

        test_loss /= len(test_eeg_loader)  # loss function already averages over batch size
        acc = correct.data.numpy() / len(test_eeg_loader.dataset)
      
        if acc > best_acc:
            best_acc = acc
            best_label = predict_label_all
            best_dis_model = copy.deepcopy(encoder_eye)
        print("Epoch: {} |training loss: {} |training acc eye: {}  |training acc eeg: {} |test loss: {} |test accuracy: {}".format(epoch, avg_closs, acc_train_eye,acc_train_eeg,avg_test_loss, acc))
        logging.info("Epoch: {} |training loss: {} |training acc eye: {}  |training acc eeg: {} |test loss: {} |test accuracy: {}".format(epoch, avg_train_loss,acc_train_eye,acc_train_eeg, avg_test_loss, acc))
    return best_acc, best_label,best_dis_model

def train():
    # load data and prepare the result dict
    cur_results_list = os.listdir(RESULT_PATH)
    cur_results_list.sort()
    cur_name_list = cur_results_list
    for i in range(len(cur_results_list)):
        item = cur_results_list[i]
        name = item.split(".")[0].split('_')[0]
        cur_name_list[i] = name
    for person_folder in person_names:
        for session in session_numbers:
            cur_name = str(person_folder)+str(session)
            if cur_name in cur_name_list:
                continue
            print(session,person_folder)
            test_res_experiment = {}
            logging.basicConfig(filename = os.path.join(log_file_name, str("cross_modality_2_")+str(person_folder)+str(session)), level=logging.DEBUG)
            logging.info("end time: ")
            logging.info(datetime.datetime.now())
            alpha = args.alpha
            common_dim = args.common_dim
            wd = args.wd
            LR = args.lr
            depth = args.num_layers
            heads = args.num_heads
            logging.critical('Current hyper-parameters of {}: |weight_decay=: {} |layers= {} |LR= {} |common_dim= {} |alpha={} |heads={}'.format(str(person_folder)+str(session),\
                wd, depth, LR ,common_dim, alpha,heads))
            
            train_data_eeg, train_label_eeg, test_data_eeg, test_label_eeg,train_data_eye, test_data_eye = splitTrainTest(args.dataset, person_folder,session,0)
            assert train_data_eeg.shape[0] == train_data_eye.shape[0]
            assert test_data_eeg.shape[0] == test_data_eye.shape[0]
            test_eeg_ol, train_eeg_ol = overlapDataForTimeWindow(test_data_eeg, train_data_eeg, time_window)
            test_eye_ol, train_eye_ol = overlapDataForTimeWindow(test_data_eye, train_data_eye, time_window)
            common_dim_tranf = int(common_dim/(time_window+1))
            test_acc,pred_label,dis_model = model(train_eeg_ol, test_eeg_ol, train_eye_ol, test_eye_ol,\
            train_label_eeg, test_label_eeg, alpha, common_dim, wd,LR,common_dim_tranf, depth, heads)
            logging.critical("best acc of all epoch of current parameter set: {}".format(test_acc))
                                  

            test_res_experiment['acc'] = test_acc
            test_res_experiment['weight_decay'] = wd
            test_res_experiment['learning_rate'] = LR
            test_res_experiment['common_dim'] = common_dim        
            test_res_experiment['depth'] = depth   
            test_res_experiment['heads'] = heads  
            test_res_experiment['pred_label'] = pred_label   
            torch.save(dis_model,MODEL_ROOT+str(person_folder)+str(session)+MODEL_NAME)


            print("Experiment {} has best acc {},  weight_decay {} and LR {} ".format(
                person_folder+'_'+session, test_res_experiment['acc']
                ,test_res_experiment['weight_decay'], test_res_experiment['learning_rate']))
            logging.info("Experiment {} has best acc {}, weight_decay {} ,common_dim {}, depth {}, heads {} and LR {} ".format(\
                person_folder+'_'+session, test_res_experiment['acc']\
                ,test_res_experiment['weight_decay'], test_res_experiment['common_dim'],\
                    test_res_experiment['depth'],test_res_experiment['heads'],test_res_experiment['learning_rate']))
            pickle.dump(test_res_experiment, open( os.path.join(RESULT_PATH, str(person_folder)+str(session)+'_cross_modality2_result.npy'), 'wb'  ))
    print("end time: ")
    print(datetime.datetime.now())
    logging.info("end time: ")
    logging.info(datetime.datetime.now())


def train_threeFold():
    # this function is for seed-v
    cur_results_list = os.listdir(RESULT_PATH)
    cur_results_list.sort()
    cur_name_list = cur_results_list
    for i in range(len(cur_results_list)):
        item = cur_results_list[i]
        name = item.split(".")[0].split('_')[0]
        cur_name_list[i] = name
    for person_folder in person_names:
        for session in session_numbers:
            cur_name = str(person_folder)+str(session)
            if cur_name in cur_name_list:
                continue
            print(session,person_folder)
            test_res_experiment = {}
            logging.basicConfig(filename = os.path.join(log_file_name, str("cross_modality_2_")+str(person_folder)+str(session)), level=logging.DEBUG)
            logging.info("end time: ")
            logging.info(datetime.datetime.now())
            alpha = args.alpha
            common_dim = args.common_dim
            wd = args.wd
            LR = args.lr
            depth = args.num_layers
            heads = args.num_heads
            logging.critical('Current hyper-parameters of {}: |weight_decay=: {} |layers= {} |LR= {} |common_dim= {} |alpha={} |heads={}'.format(str(person_folder)+str(session),\
                wd, depth, LR ,common_dim, alpha,heads))
            test_acc_list = []
            train_acc_list = []
            model_list = []
            pre_label_list = []
            for fold in range(3):
            #person_folder, session, path = oneExperiment("EEG")
                num_experiment = 0     
                logging.critical('Current hyper-parameters of {}: |weight_decay=: {} |layers= {} |LR= {} |common_dim= {} |alpha={} |heads={}'.format(str(person_folder),\
                    wd, depth, LR ,common_dim, alpha,heads))
                train_data_eeg, train_label_eeg, test_data_eeg, test_label_eeg,train_data_eye, test_data_eye = splitTrainTest(args.dataset, person_folder,session,fold)
                assert train_data_eeg.shape[0] == train_data_eye.shape[0]
                assert test_data_eeg.shape[0] == test_data_eye.shape[0]
                test_eeg_ol, train_eeg_ol = overlapDataForTimeWindow(test_data_eeg, train_data_eeg, time_window)
                test_eye_ol, train_eye_ol = overlapDataForTimeWindow(test_data_eye, train_data_eye, time_window)
                common_dim_tranf = int(common_dim/(time_window+1))
                test_acc,pred_label,dis_model = model(train_eeg_ol, test_eeg_ol, train_eye_ol, test_eye_ol,\
                train_label_eeg, test_label_eeg, alpha, common_dim, wd,LR,common_dim_tranf, depth, heads)
                logging.critical("best acc of all epoch of current parameter set: {}".format(test_acc))
                num_experiment = num_experiment + 1
                test_acc_list.append(test_acc)
                pre_label_list.append(pred_label)
                model_list.append(dis_model)
           
                                  

            test_res_experiment['acc'] = np.mean(test_acc_list)
            test_res_experiment['weight_decay'] = wd
            test_res_experiment['learning_rate'] = LR
            test_res_experiment['common_dim'] = common_dim        
            test_res_experiment['depth'] = depth   
            test_res_experiment['heads'] = heads  
            test_res_experiment['pred_label'] = pre_label_list   
            torch.save(model_list,MODEL_ROOT+str(person_folder)+str(session)+MODEL_NAME)


            print("Experiment {} has best acc {},  weight_decay {} and LR {} ".format(
                person_folder+'_'+session, test_res_experiment['acc']
                ,test_res_experiment['weight_decay'], test_res_experiment['learning_rate']))
            logging.info("Experiment {} has best acc {}, weight_decay {} ,common_dim {}, depth {}, heads {} and LR {} ".format(\
                person_folder+'_'+session, test_res_experiment['acc']\
                ,test_res_experiment['weight_decay'], test_res_experiment['common_dim'],\
                    test_res_experiment['depth'],test_res_experiment['heads'],test_res_experiment['learning_rate']))
            pickle.dump(test_res_experiment, open( os.path.join(RESULT_PATH, str(person_folder)+str(session)+'_cross_modality2_result.npy'), 'wb'  ))
    print("end time: ")
    print(datetime.datetime.now())
    logging.info("end time: ")
    logging.info(datetime.datetime.now())


if __name__ == "__main__":
    if args.dataset == 'seedv':
        train_threeFold()
    else:
        train()