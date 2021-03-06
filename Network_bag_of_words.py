from __future__ import print_function
import tensorflow as tf
import numpy as np
import pickle
import batch_generator



class EndToEndNetwork:
    def __init__(self, num_hops=1, batch_size=100, validation_size=400, sentence_input_embed_dim=200,
                 sentence_output_embed_dim=200):
        self.batch_generator = batch_generator.BatchGenerator(batch_size)
        self.num_hops = num_hops
        self.question_embed_dim = sentence_input_embed_dim
        self.sentence_input_embed_dim = sentence_input_embed_dim
        self.sentence_ouput_embed_dim = sentence_output_embed_dim
        self.vocabluary_size = len(self.batch_generator.encode_dict)
        self.memory_dim = sentence_output_embed_dim
        self.batch_size = batch_size
        self.validation_size = validation_size
        self.session = tf.Session()
        self.bb = tf.Variable(tf.truncated_normal([1, self.question_embed_dim], stddev=0.1))
        self.ba = tf.Variable(tf.truncated_normal([1, self.sentence_input_embed_dim], stddev=0.1))
        self.bc = tf.Variable(tf.truncated_normal([1, self.sentence_ouput_embed_dim], stddev=0.1))
        self.A = []
        self.B = tf.Variable(tf.truncated_normal([self.vocabluary_size, self.question_embed_dim], stddev=0.1))
        self.C = []
        self.W = tf.Variable(tf.truncated_normal([self.sentence_ouput_embed_dim, self.vocabluary_size], stddev=0.1))
        for i in range(self.num_hops):
            self.A.append(
                tf.Variable(tf.truncated_normal([self.vocabluary_size, self.sentence_input_embed_dim], stddev=0.1)))
            self.C.append(
                tf.Variable(tf.truncated_normal([self.vocabluary_size, self.sentence_ouput_embed_dim], stddev=0.1)))
        self.init = tf.initialize_all_variables()

    def init_variables(self):
        self.session.run(self.init)

    def build_graph(self):
        # Variables
        A = self.A
        B = self.B
        C = self.C
        W = self.W
        bb = self.bb
        ba = self.ba
        bc = self.bc

        # Batch Graph
        batch_question = tf.placeholder(tf.int32, [None, None])
        batch_sentences = tf.placeholder(tf.int32, [None, None, None])
        batch_y_ = tf.placeholder(tf.int32, [None])

        cross_entropy = tf.zeros([])
        batch_y_one_hot = tf.one_hot(batch_y_, self.batch_generator.num_variants)
        global_step = tf.Variable(0, trainable=False)
        self.make_step = global_step.assign(global_step + 1)
        starter_learning_rate = 0.0001
        learning_rate = starter_learning_rate
        # learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 350, 0.5, staircase=False)

        for i in range(self.batch_size):
            question = batch_question[i, :]
            sentences = batch_sentences[i, :, :]
            y_ = tf.reshape(batch_y_one_hot[i, :], [1, self.batch_generator.num_variants])

            question_embed = tf.reshape(
                tf.reduce_sum(
                    tf.nn.embedding_lookup(B, question), 0),
                [1, self.question_embed_dim]) + bb

            u = question_embed

            for j in range(self.num_hops):
                sentences_output_embed = tf.reduce_sum(
                    tf.nn.embedding_lookup(C[j], sentences), 1) + bc
                sentences_input_embed = tf.reduce_sum(
                    tf.nn.embedding_lookup(A[j], sentences), 1) + ba

                p = tf.nn.softmax(
                    tf.matmul(u, sentences_input_embed, transpose_b=True))
                u = u + tf.reduce_sum(
                    tf.mul(tf.transpose(p), sentences_output_embed), 0)

            y = p

            cross_entropy = cross_entropy - tf.reduce_sum(y_ * tf.log(y))

        self.learning_rate = learning_rate
        #         train_step = (
        #             tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)
        #         )

        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        gvs = optimizer.compute_gradients(cross_entropy)
        grads = tf.clip_by_global_norm([grad for grad, var in gvs], 40)
        self.grad_before_clipping = grads[1]
        self.grad_after_clipping = tf.clip_by_global_norm(grads[0], 1)[1]
        capped_gvs = [(grads[0][i], gvs[i][1]) for i in
                      range(len(gvs))]  # [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        self.app_grads = optimizer.apply_gradients(capped_gvs)

        #        train_step = tf.train.AdagradOptimizer(0.01).minimize(cross_entropy)

        # Validation Graph
        validation_question = tf.placeholder(tf.int32, [None, None])
        validation_sentences = tf.placeholder(tf.int32, [None, None, None])
        validation_y_ = tf.placeholder(tf.int32, [None])
        validation_y_one_hot = tf.one_hot(validation_y_, self.batch_generator.num_variants)

        correct_rate = tf.zeros([])
        predictions_list = []
        support_list = []

        for i in range(self.validation_size):
            question = validation_question[i, :]
            sentences = validation_sentences[i, :, :]
            y_ = tf.reshape(validation_y_one_hot[i, :], [1, self.batch_generator.num_variants])

            question_embed = tf.reshape(
                tf.reduce_sum(
                    tf.nn.embedding_lookup(B, question), 0),
                [1, self.question_embed_dim])

            u = question_embed

            for j in range(self.num_hops):
                sentences_output_embed = tf.reduce_sum(
                    tf.nn.embedding_lookup(C[j], sentences), 1)
                sentences_input_embed = tf.reduce_sum(
                    tf.nn.embedding_lookup(A[j], sentences), 1)

                p = tf.nn.softmax(
                    tf.matmul(u, sentences_input_embed, transpose_b=True))
                support_list.append(tf.argmax(p, 1))
                u = u + tf.reduce_sum(
                    tf.mul(tf.transpose(p), sentences_output_embed), 0)

            y = p

            predictions_list.append(tf.argmax(y, 1))

        support = tf.reshape(
            tf.pack(support_list), [self.validation_size, self.num_hops])
        predictions = tf.reshape(tf.pack(predictions_list), [-1])
        correct_rate = tf.reduce_mean(
            tf.cast(
                tf.equal(
                    predictions, tf.argmax(validation_y_one_hot, 1)),
                tf.float32))

        self.batch_placeholder = [batch_sentences, batch_question, batch_y_]
        self.validation_placeholder = [validation_sentences, validation_question, validation_y_]
        #        self.train_step = train_step
        self.correct_rate = correct_rate
        self.predictions = predictions
        self.support = support
        self.cross_entropy = cross_entropy
        self.y = y
        self.init = tf.initialize_all_variables()

    def train(self, steps):
        for i in range(steps):
            batch = self.batch_generator.get_text_batch(encode=True, batch_type='train')
            # batch = self.batch
            batch_dic = {self.batch_placeholder[0]: batch[2], self.batch_placeholder[1]: batch[3],
                         self.batch_placeholder[2]: batch[4]}
            #            if i % 100 == 0:
            #                print(self.session.run(self.learning_rate, feed_dict = batch_dic), " ", self.session.run(self.grad_before_clipping, feed_dict = batch_dic), self.session.run(self.grad_after_clipping, feed_dict = batch_dic))
            self.session.run(self.app_grads, feed_dict=batch_dic)
            self.session.run(self.make_step, feed_dict=batch_dic)

    def validate(self, print_examples, print_score=True):
        validation_set = self.batch_generator.get_text_batch(encode=True, batch_size=self.validation_size, batch_type='validation')
        # validation_set = self.batch
        validation_dict = {self.validation_placeholder[0]: validation_set[2],
                           self.validation_placeholder[1]: validation_set[3],
                           self.validation_placeholder[2]: validation_set[4]}
        if print_score == True:
            print("Correct rate:  ", self.session.run(self.correct_rate, feed_dict=validation_dict))
        ans = self.session.run(self.predictions, feed_dict=validation_dict)
        sup = self.session.run(self.support, feed_dict=validation_dict)
        y = self.session.run(self.y, feed_dict=validation_dict)
        for x in np.random.random_integers(0, self.validation_size - 1, print_examples):
            print(validation_set[1][x], validation_set[0][x])
            # preprocessing.print_task(validation_set[0][[x], :, :], validation_set[1][[x], :], validation_set[2][[x], :],
            #                          self.batch_gen.words)
            print("\n")
            print("Model's answer: ", end=" ")
            print(ans[x])
            # print("Support sentences:")
            # indicies = list(sup[x, :])
            # preprocessing.print_sentences(np.reshape(validation_set[0][x, indicies, :], [1, self.num_hops, -1]),
            #                               self.batch_gen.words)
            #
            # print("-------------------------")
        return self.session.run(self.correct_rate, feed_dict=validation_dict)

def train_yahoo():
    n = EndToEndNetwork()
    n.build_graph()
    n.init_variables()
    n.train(700)
    n.validate(2)

train_yahoo()
