from tensorflow import keras
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import optimizers, callbacks, layers, losses
from tensorflow.keras.layers import Dense, Concatenate, Activation, Add, BatchNormalization, Dropout, Input, Embedding, Flatten, Multiply, Dot
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.utils import to_categorical


class BPR_Triplet(keras.Model):
    def __init__(self, u_dim, i_dim, latent_dim):
        super(BPR_Triplet, self).__init__()
        
        self.u_dim = u_dim
        self.i_dim = i_dim
        self.latent_dim = latent_dim
        
        self.model = self.build_model()

    def compile(self, optim):
        super(BPR_Triplet, self).compile()
        self.optim = optim
    
    def build_model(self):
        u_input = Input(shape=(1, ))
        i_input = Input(shape=(1, ))

        u_emb = Flatten()(Embedding(self.u_dim, self.latent_dim, input_length=u_input.shape[1])(u_input))
        i_emb = Flatten()(Embedding(self.i_dim, self.latent_dim, input_length=i_input.shape[1])(i_input))

        mul = Dot(1)([u_emb, i_emb])

#         out = Dense(1)(mul)
        
        return Model([u_input, i_input], mul)
    
    def train_step(self, data):
        user, pos, neg = data[0]

        with tf.GradientTape() as tape:
            pos_d = self.model([user, pos])
            neg_d = self.model([user, neg])
            
            loss = -tf.reduce_mean(tf.math.log(tf.sigmoid(pos_d - neg_d)))

        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optim.apply_gradients(zip(grads, self.model.trainable_weights))
        
        return {'loss': loss}
    
    def call(self, data):
        user, item = data
        return self.model([user, item])