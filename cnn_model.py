from python.util import *
from python.data_helper import *

vocab_list_file     = 'data/word_list.txt'
train_data_file     = 'data/sentence_train.txt'
label_data_file     = 'data/sentence_answer.txt'
log_file            = 'log/rnn_training_v25'
save_file_name      = 'model/rnn_training_v25'

                        #[ (fitter_size, kernal_size)] 
cnn_fitter_kernal_size = [ (12, 5) ]
fc_layer_size          = [ 32 ]

learning_rate = 1e-3
embedding_dim = 64
seq_length    = 12

helper    = Data_Helper(train_data_file, label_data_file, vocab_list_file)
vocab_size= helper.get_vocab_size()

input_x = tf.placeholder(tf.int32, [None, seq_length], name='input_x')
input_y = tf.placeholder(tf.int64, [None, 4], name='input_y')
dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

transpose_y = tf.transpose(input_y)

y           = transpose_y[0]
adverb_y    = transpose_y[1]
object_y    = transpose_y[2]
number_y    = transpose_y[3]

with tf.device('/cpu:0'):
    embedding = tf.get_variable('embedding', [vocab_size, embedding_dim])
    embedding_inputs = tf.nn.embedding_lookup(embedding, input_x)

cnn_model = make_cnn_model(embedding_inputs, cnn_fitter_kernal_size, fc_layer_size, dropout_prob)

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

global_step   = tf.train.get_or_create_global_step()
increment_global_step = tf.assign(global_step, global_step + 1)

init = tf.initialize_all_variables()
sess = tf.Session()
#with tf.Session() as sess:

sess.run(init)

def train_model(iter_size=1000, batch_size=50, show_step=10, test_step=100):
    for i in range(iter_size):
        train_data, label_data = helper.get_data(batch_size, pad_sequence_length=seq_length)
        out = sess.run([global_loss, acc_rate, adverb_rate, object_rate, number_rate, increment_global_step, opt], { input_x : train_data, input_y: label_data, dropout_prob : 0.5})
        if out[5] % show_step == 0:
                tf.logging.info('Step #%d: accuracy %.2f%%, cross entropy %f' %
                    (out[5], (out[1] + out[2] + out[3] + out[4]) * 25,
                     out[0]))


def test_model(test_list):
    source_data, processed_data = helper.string_to_input_data(test_list, pad_sequence_length=seq_length)
    out_data = sess.run([pred_class, adverb_class, object_class, number_class], { input_x : processed_data, dropout_prob : 1})
    return get_output_string(out_data, source_data)

train_model()
