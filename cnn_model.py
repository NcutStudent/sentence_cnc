from python.util import *
from python.data_helper import *

vocab_list_file     = 'data/word_list.txt'
train_data_file     = 'data/sentence_train.txt'
label_data_file     = 'data/sentence_answer.txt'
log_file            = 'log/rnn_training_v25'
save_file_name      = 'model/rnn_training_v25'


cnn_fitters_list = []
cnn_kernal_list  = []

learning_rate = 1e-3
embedding_dim = 128
seq_length    = 12

helper    = Data_Helper(train_data_file, label_data_file, vocab_list_file)
vocab_size= helper.get_vocab_size()

input_x = tf.placeholder(tf.int32, [None, seq_length], name='input_x')
input_y = tf.placeholder(tf.int64, [None, 4], name='input_y')

transpose_y = tf.transpose(input_y)

y           = transpose_y[0]
adverb_y    = transpose_y[1]
object_y    = transpose_y[2]
number_y    = transpose_y[3]

with tf.device('/cpu:0'):
    embedding = tf.get_variable('embedding', [vocab_size, embedding_dim])
    embedding_inputs = tf.nn.embedding_lookup(embedding, input_x)

cnn_model = make_cnn_model(embedding_inputs)

pred_vec      = tf.layers.dense(cnn_model, seq_length, name='pred_vector')
adverb_vec    = tf.layers.dense(cnn_model, seq_length, name='pred_adverb_vector')
object_vec    = tf.layers.dense(cnn_model, seq_length, name='pred_object_vector')
number_vec    = tf.layers.dense(cnn_model, seq_length, name='pred_number_vector')

pred_class    = tf.argmax(tf.nn.softmax(pred_vec), 1, name='pred_class')
adverb_class  = tf.argmax(tf.nn.softmax(adverb_vec), 1, name='adverb_class')
object_class  = tf.argmax(tf.nn.softmax(object_vec), 1, name='object_class')
number_class  = tf.argmax(tf.nn.softmax(number_vec), 1, name='number_class')

loss          = make_loss(pred_vec, y, seq_length, name='loss')
adverb_loss   = make_loss(adverb_vec, adverb_y, seq_length, name='adverb_loss')
object_loss   = make_loss(object_vec, object_y, seq_length, name='object_loss')
number_loss   = make_loss(number_vec, number_y, seq_length, name='number_loss')

acc_rate      = make_acc_rate(pred_class, y, name='acc_rate')
adverb_rate   = make_acc_rate(adverb_class, adverb_y, name='adverb_loss')
object_rate   = make_acc_rate(object_class, object_y, name='object_loss')
number_rate   = make_acc_rate(number_class, number_y, name='number_loss')

global_loss   = loss + adverb_loss + object_loss + number_loss

opt      = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(global_loss, name='optimizer')

init = tf.initialize_all_variables()
sess = tf.Session()
#with tf.Session() as sess:
sess.run(init)
for i in range(100):
    train_data, label_data = helper.get_data(100, pad_sequence_length=seq_length)
    loss_out, opt_out = sess.run([global_loss, opt], { input_x : train_data, input_y: label_data })
    print(loss_out)
