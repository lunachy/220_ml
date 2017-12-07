# coding=utf-8
import logging.handlers
import os
import time
import sys
from multiprocessing import cpu_count, Pool
import pickle as cPickle
import numpy as np
import pandas as pd
import pefile
import signal
from malware_modeling import Vocab
import scipy.misc
from functools import reduce
from itertools import chain
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

IMP_NAMES = []
batch_size = 16
epochs = 50
train_size = 40000
train_size_2 = 47059
MAXLEN = 10000
OUTPUT_DIM = 50
#CPU_COUNT = cpu_count()  # 10
CPU_COUNT = 10


#root_dir = "F:/virus"
root_dir = "/data/root/pe_classify/"
root_dir_2 = "/root/pe_classify/"
pefile_train_dir = os.path.join(root_dir, '2017game_train')
pefile_test_dir = os.path.join(root_dir, '2017game_test')
asm_train_dir = os.path.join(root_dir, '2017game_train_asm')
asm_test_dir = os.path.join(root_dir, '2017game_test_asm')
pefile_unpack_train_dir = os.path.join(root_dir_2, 'unpack_train')
pefile_unpack_test_dir = os.path.join(root_dir_2, 'unpack_test')
asm_unpack_train_dir = os.path.join(root_dir_2, 'unpack_train_asm')
asm_unpack_test_dir = os.path.join(root_dir_2, 'unpack_test_asm')

imp_name_uncrypt_path = os.path.join(root_dir, 'imp_names_uncrypt_map.dat')
imp_name_path = os.path.join(root_dir, 'imp_names_map.dat')

ops_x_train_path = os.path.join(root_dir, 'ops_new.npz')
imp_x_train_path = os.path.join(root_dir, 'imp_new.npz')
ops_x_test_path = os.path.join(root_dir, 'ops_test_new.npz')
imp_x_test_path = os.path.join(root_dir, 'imp_test_new.npz')
ops_uncrypt_train_path = os.path.join(root_dir, 'ops_uncrypt_train.npz')
ops_uncrypt_test_path = os.path.join(root_dir, 'ops_uncrypt_test.npz')
imp_uncrypt_train_path = os.path.join(root_dir, 'imp_uncrypt_train.npz')
imp_uncrypt_test_path = os.path.join(root_dir, 'imp_uncrypt_test.npz')
ops_combined_train_path = os.path.join(root_dir, 'ops_combined_train.npz')
ops_combined_test_path = os.path.join(root_dir, 'ops_combined_test.npz')
imp_combined_train_path = os.path.join(root_dir, 'imp_combined_train.npz')
imp_combined_test_path = os.path.join(root_dir, 'imp_combined_test.npz')
ops_3g_train_path = os.path.join(root_dir, 'ops_3g_train.npz')
ops_3g_test_path = os.path.join(root_dir, 'ops_3g_test.npz')
ops_4g_train_path = os.path.join(root_dir, 'ops_4g_train.npz')
ops_4g_test_path = os.path.join(root_dir, 'ops_4g_test.npz')


train_csv = os.path.join(root_dir, '2017game_train.csv')
test_csv = os.path.join(root_dir, '2017game_test.csv')
train_uncrypt_csv = os.path.join(root_dir, 'train_uncrypt.csv')
test_uncrypt_csv = os.path.join(root_dir, 'test_uncrypt.csv')
train_md5_sig_encoding_path = os.path.join(root_dir, 'train_md5_sig_encoding.npz')
test_md5_sig_encoding_path = os.path.join(root_dir, 'test_md5_sig_encoding.npz')
train_md5_api_encoding_path = os.path.join(root_dir, 'train_md5_api_encoding.npz')
test_md5_api_encoding_path = os.path.join(root_dir, 'test_md5_api_encoding.npz')



log = logging.getLogger()
formatter = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")

fh = logging.handlers.WatchedFileHandler(os.path.join(root_dir, 'pe_analyzer.log'))
fh.setFormatter(formatter)
log.addHandler(fh)
ch = logging.StreamHandler()
ch.setFormatter(formatter)
log.addHandler(ch)

log.setLevel(logging.INFO)


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def filter_array(ops):
    return filter(lambda op:op!=1,ops)

def create_ngram_set(input_list, ngram_value=2):
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))

def add_ngram(sequences, token_indice, ngram_range=2):
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for i in range(len(new_list) - ngram_range + 1):
            for ngram_value in range(2, ngram_range + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)
    return new_sequences


def most_common(x):
    x2=Counter(x).most_common(1)
    return x2[0][0]


####################full data####################################
train_data = pd.read_csv(train_csv)
label_dic = dict(zip(train_data["md5"], train_data["type"]))
asm_files_train = os.listdir(asm_train_dir)
md5s_train = [md5 for md5 in train_data['md5'] if (md5 + ".asm") in asm_files_train]
# train_label = train_label[train_label['md5'].isin(md5s_train)]
train_label = [label_dic[md5] for md5 in md5s_train]
test_data = pd.read_csv(test_csv)
label_dic_test = dict(zip(test_data["md5"], test_data["type"]))
asm_files_test = os.listdir(asm_test_dir)
md5s_test = [md5 for md5 in test_data['md5'] if (md5 + ".asm") in asm_files_test]
test_label = [label_dic_test[md5] for md5 in md5s_test]
print(len(md5s_train), len(md5s_test))
print(len(train_label), len(test_label))
#
#####y label###############
nb_class = len(set(train_label))
label = sorted(set(train_label))
dic = dict(zip(label, range(0, nb_class)))
y_combined = [dic[y] for y in train_label]
#y_combined = to_categorical(y_combined, nb_class)
y_test_combined = [dic[y] for y in test_label]
#y_test_combined = to_categorical(y_test_combined, nb_class)
#print(y_combined.shape,y_test_combined.shape)
print(len(y_combined),len(y_test_combined))


# ############load imp data####################
imp_x_train_combined = np.load(imp_x_train_path)["arr_0"]
imp_x_train_combined = np.array(imp_x_train_combined)
imp_x_test_combined = np.load(imp_x_test_path)["arr_0"]
imp_x_test_combined = np.array(imp_x_test_combined)
print(imp_x_train_combined.shape, imp_x_test_combined.shape)
width=167 #27889
#width = 161
imp_x_train_combined_flat = np.reshape(imp_x_train_combined, (imp_x_train_combined.shape[0], width * width))
imp_x_test_combined_flat = np.reshape(imp_x_test_combined ,(imp_x_test_combined .shape[0],width*width))
print(imp_x_train_combined_flat.shape, imp_x_test_combined_flat.shape)
#

##########load ops_3g data##################
ops_x_train_3g = np.load(ops_3g_train_path)["arr_0"]
ops_x_train_3g = np.array(ops_x_train_3g)
ops_x_test_3g = np.load(ops_3g_test_path)["arr_0"]
ops_x_test_3g = np.array(ops_x_test_3g)
print(ops_x_train_3g.shape, ops_x_test_3g.shape)

#########load sandbox sig data ##########
train_md5s = np.load(train_md5_sig_encoding_path)["md5"]
train_sigs_encoding = np.load(train_md5_sig_encoding_path)["sig"]
test_md5s = np.load(test_md5_sig_encoding_path)["md5"]
test_sigs_encoding = np.load(test_md5_sig_encoding_path)["sig"]
#### rearrange the order
sigs_dic = dict(zip(train_md5s, train_sigs_encoding))
train_sigs_encoding_re = [sigs_dic[md5] for md5 in md5s_train]
train_sigs_encoding_re = np.array(train_sigs_encoding_re)
sigs_dic_test = dict(zip(test_md5s, test_sigs_encoding))
test_sigs_encoding_re = [sigs_dic_test[md5] for md5 in md5s_test]
test_sigs_encoding_re = np.array(test_sigs_encoding_re)
print(train_sigs_encoding_re.shape, test_sigs_encoding_re.shape)

#########load sandbox apistat data ##########
train_md5s=np.load(train_md5_api_encoding_path)["md5"]
train_api_encoding = np.load(train_md5_api_encoding_path)["api"]
test_md5s=np.load(test_md5_api_encoding_path)["md5"]
test_api_encoding=np.load(test_md5_api_encoding_path)["api"]
#### rearrange the order
api_dic = dict(zip(train_md5s, train_api_encoding))
train_api_encoding_re = [api_dic[md5] for md5 in md5s_train]
train_api_encoding_re = np.array(train_api_encoding_re)
api_dic_test = dict(zip(test_md5s, test_api_encoding))
test_api_encoding_re = [api_dic_test[md5] for md5 in md5s_test]
test_api_encoding_re = np.array(test_api_encoding_re)
print(train_api_encoding_re.shape, test_api_encoding_re.shape)


##### rf for apistat #####################
X_train = train_api_encoding_re
X_test = test_api_encoding_re
y_train = y_combined
y_test = y_test_combined
print(X_train.shape,X_test.shape,len(y_train),len(y_test))
#((47059, 297), (9398, 297), 47059, 9398)
clf = RandomForestClassifier(n_jobs=4)
clf.fit(X_train, y_train)
# sum
y_pred_api=clf.predict(X_test)
print(np.mean(y_pred_api == y_test)) ## 0.620025537348
# rf_api_path=os.path.join(root_dir, '0926_rf_api.pkl')
# with open(rf_api_path, 'wb') as f:
#     cPickle.dump(clf, f)
# with open(rf_api_path, 'rb') as f:
#     clf = cPickle.load(f)

##### rf for sandbox(sig+apistat) #####################
x_train_sb=np.concatenate([train_api_encoding_re,train_sigs_encoding_re],axis=1)
x_test_sb=np.concatenate([test_api_encoding_re,test_sigs_encoding_re],axis=1)
X_train=x_train_sb
X_test=x_test_sb
y_train=y_combined
y_test=y_test_combined
print(X_train.shape,X_test.shape,len(y_train),len(y_test))
#((47059, 476), (9398, 476), 47059, 9398)
clf = RandomForestClassifier(n_jobs=4)
clf.fit(X_train, y_train)
y_pred_sb=clf.predict(X_test)
print(np.mean(y_pred_sb==y_test)) ##0.636837625027
# rf_sandbox_path=os.path.join(root_dir, '0926_rf_sandbox.pkl')
# with open(rf_sandbox_path, 'wb') as f:
#     cPickle.dump(clf, f)
# with open(rf_sandbox_path, 'rb') as f:
#     clf = cPickle.load(f)


##### rf for ops_3g #####################
X_train = ops_x_train_3g
X_test = ops_x_test_3g
y_train = y_combined
y_test = y_test_combined
print(X_train.shape,X_test.shape,len(y_train),len(y_test))
#((47059, 30380), (9398, 30380), 47059, 9398)
clf = RandomForestClassifier(n_jobs=4)
clf.fit(X_train, y_train)
y_pred_ops = clf.predict(X_test)
print(np.mean(y_pred_ops == y_test)) ##0.610129814854
# rf_ops_path=os.path.join(root_dir, '0926_rf_ops.pkl')
# with open(rf_ops_path, 'wb') as f:
#     cPickle.dump(clf, f)
# with open(rf_ops_path, 'rb') as f:
#     clf = cPickle.load(f)


##### rf for ops_3g+sandbox(sig+apistat) #####################
x_train_combined_3 = np.concatenate([x_train_sb,ops_x_train_3g],axis=1)
x_test_combined_3 = np.concatenate([x_test_sb,ops_x_test_3g],axis=1)
X_train=x_train_combined_3
X_test=x_test_combined_3
y_train=y_combined
y_test=y_test_combined
print(X_train.shape,X_test.shape,len(y_train),len(y_test))
#((47059, 30856), (9398, 30856), 47059, 9398)
clf = RandomForestClassifier(n_jobs=4)
clf.fit(X_train, y_train)
y_pred_sops = clf.predict(X_test)
print(np.mean(y_pred_sops == y_test)) ## 0.664928708236
# rf_sops_path=os.path.join(root_dir, '0926_rf_sops.pkl')
# with open(rf_sops_path, 'wb') as f:
#     cPickle.dump(clf, f)
# with open(rf_sops_path, 'rb') as f:
#     clf = cPickle.load(f)


##### rf for ops_3g+imp+sandbox(sig+apistat) #####################
x_train_combined = np.concatenate([x_train_sb,ops_x_train_3g,imp_x_train_combined_flat],axis=1)
x_test_combined = np.concatenate([x_test_sb,ops_x_test_3g,imp_x_test_combined_flat],axis=1)
X_train = x_train_combined
X_test = x_test_combined
y_train = y_combined
y_test = y_test_combined
print(X_train.shape,X_test.shape,len(y_train),len(y_test))
#((47059, 58745), (9398, 58745), 47059, 9398)
clf = RandomForestClassifier(n_jobs=4)
clf.fit(X_train, y_train)
y_pred_all4=clf.predict(X_test)
print(np.mean(y_pred_all4 == y_test)) ##0.668546499255
# rf_all4_path=os.path.join(root_dir, '0926_rf_sandbox_ops_imp.pkl')
# with open(rf_all4_path, 'wb') as f:
#     cPickle.dump(clf, f)
# with open(rf_all4_path, 'rb') as f:
#     clf = cPickle.load(f)

### ensemble #################
y_pred_total=[]
y_pred_total.append(y_pred_api)
y_pred_total.append(y_pred_sb)
y_pred_total.append(y_pred_ops)
y_pred_total.append(y_pred_sops)
y_pred_total.append(y_pred_all4)
y_pred_total = pd.DataFrame(y_pred_total).T
y_pred_total.columns=["api","sandbox","ops","sandbox_ops","sandbox_ops_imp"]
y_pred_total.to_csv("/data/root/pe_classify/092602.csv")
y_pred = map(most_common, np.array(y_pred_total))
np.mean(np.array(y_pred) == y_test)  ##0.70366035326665244