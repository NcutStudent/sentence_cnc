import tensorflow as tf
import jieba
import re

jieba.load_userdict("data/user_direct.txt")


def get_vocab_data(filename, encoding='utf-8'):
    f = open(filename, encoding=encoding)
    d = {}
    for l in f:
        name, i = l.split(' ')
        d[name] = int(i)
    return d


def make_cnn_model(x, cnn_list=[(128, 5)], fc_hidden_list=[64], dropout=0.5):
    out=x
    i = 1

    for fitter_size, kernel_size in cnn_list:
        out = tf.layers.conv1d(out, fitter_size, kernel_size, name='conv' + str(i))

    out = tf.reduce_max(out, reduction_indices=[1], name='gmp')

    i = 0
    for hidden_dim in fc_hidden_list:
        out = tf.layers.dense(out, hidden_dim, name='fc' + str(i))
        out = tf.contrib.layers.dropout(out, dropout)
        out = tf.nn.relu(out)
    
    return out

def make_acc_rate(pred_class, input_y, name=None):
    correct_count = tf.equal(input_y, pred_class)
    acc_rate      = tf.reduce_mean(tf.cast(correct_count, tf.float32), name=name)
    return acc_rate

def make_loss(pred_vec, input_y, seq_length, name=None):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred_vec, labels=tf.one_hot(input_y, seq_length))
    loss          = tf.reduce_mean(cross_entropy, name=name)
    return loss

def process_string_with_jieba(string, with_number=False):
    arr = re.sub("([\w\W\u4e00-\u9fff]-)", "", string)
    arr = re.sub("[\u3002\uff1b\uff0c\uff1a\u201c\u201d\uff08\uff09\u3001\uff1f\u300a\u300b\n]", "", arr)
    if not with_number:
        arr = re.sub("[\d]+", "NUMBERFLAG", arr)
    return jieba.cut(arr)
