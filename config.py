import os

RESULT_PATH = ""
MODEL_ROOT = ""
MODEL_NAME = ""
LOG_NAME = ""
EYE_DIM = 0
EEG_DIM = 310
OUTPUT_DIM = 0

unx_path_eeg = "/home/wangyiting/Dataset/seed/eeg_disperse"
unx_path_eye = "/home/wangyiting/Dataset/seed/eye_disperse"

unx_path_eeg_iv = "/home/wangyiting/Dataset/eeg_disperse_seed_iv"
unx_path_eye_iv = "/home/wangyiting/Dataset/eye_disperse_seed_iv"

unx_path_eeg_v = "/home/wangyiting/Dataset/eeg_disperse_seed_v/"
unx_path_eye_v = "/home/wangyiting/Dataset/eye_disperse_seed_v/"

label_mat_unx = '/home/wangyiting/Dataset/seed/label.mat'

person_names = os.listdir(unx_path_eeg)
person_names.sort()

person_names_iv = os.listdir(unx_path_eeg_iv)
person_names_iv.sort()

session_numbers = ['1','2','3']
session_numbers.sort()

session_numbers_v = ['1','2','3']
session_numbers_v.sort()

log_file_name = '/home/wangyiting/multimodal/logs/'

time_window = 5

seediv_class_num = 4


person_names_v = os.listdir(unx_path_eeg_v)
person_names_v.sort()
person_names_v_reverse = person_names_v
person_names_v_reverse.reverse()

SEEDIV_EYE_DIM = 31
SEEDV_EYE_DIM = 33
SEED_EYE_DIM = 41
