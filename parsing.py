from argparse import ArgumentParser
import os
from config import *

def ensureDirPath(path):
    if not os.path.exists(path):
        os.makedirs(path)

def parse_train_args():
    parser = ArgumentParser()

    #train
    parser.add_argument("--lr", type=float,default=1e-3)
    parser.add_argument("--wd", type=float,default=1e-4)
    parser.add_argument("--alpha",type=float,default=0.2)
    parser.add_argument("--epochs",type=int,default=500)
    
    #model
    parser.add_argument("--num_layers", type=int,default=3)
    parser.add_argument("--num_heads", type=int,default=3)
    parser.add_argument("--common_dim", type=int,default=78)

    #data
    parser.add_argument("--dataset", type=str, choices=['seed','seediv','seedv'],default='seed')

    args = parser.parse_args()

    RESULT_PATH = "./results/"+args.dataset+"/"
    MODEL_ROOT = "./model/"+args.dataset+"/"
    ensureDirPath(RESULT_PATH)
    ensureDirPath(MODEL_ROOT)

    MODEL_NAME = args.dataset+"_eye_encoder_cross_modality.pt"
    match args.dataset:
        case "seed":
            EYE_DIM = SEED_EYE_DIM
            OUTPUT_DIM = 3
        case "seediv":
            EYE_DIM = SEEDIV_EYE_DIM
            OUTPUT_DIM = 4
        case "seedv":
            EYE_DIM = SEEDV_EYE_DIM
            OUTPUT_DIM = 5
        case _:
            EYE_DIM = 0
            OUTPUT_DIM = 0

    

    return args