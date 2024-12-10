import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf

import matplotlib.pyplot as plt
import math

def network_param():
    ### Network parameters are adapted to suit my project, details explained in Model section of the github repo
    network_architecture = \
                            {'n_input':108,\
                              'n_z':100,\
                              'size_slices':[20, 16, 20, 16, 18, 18],\
                              # BECAREFUL The number of slice should be equal to the number of _mod_ network
                              'size_slices_shared':[50, 40, 50, 40, 45, 45],\
                              # BECAREFUL The sum of the dimensions of the slices, should be equal to the last dec_shared
                              'mod0':[100,50],\
                              'mod1':[80,40],\
                              'mod2':[100,50],\
                              'mod3':[80,40],\
                              'mod4':[90,45],\
                              'mod5':[90,45],\
                              'mod0_2':[100,20],\
                              'mod1_2':[80,16],\
                              'mod2_2':[100,20],\
                              'mod3_2':[80,16],\
                              'mod4_2':[90,18],\
                              'mod5_2':[90,18],\
                              'enc_shared':[350],\
                              'dec_shared':[350,270]}
    return network_architecture




class VariationalAutoencoder(object):
    """ Variation Autoencoder (VAE) with an sklearn-like interface implemented using TensorFlow.
    
    This implementation uses probabilistic encoders and decoders using Gaussian distributions and  
    realized by multi-layer perceptrons. The VAE can be learned end-to-end.
    
    See "Auto-Encoding Variational Bayes" by Kingma and Welling for more details.
    """
    def __init__(self,sess, network_architecture, transfer_fct=tf.nn.relu,  learning_rate=0.001, batch_size=100, vae_mode=False, vae_mode_modalities=False):
        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.vae_mode = vae_mode
        self.vae_mode_modalities = vae_mode_modalities

        self.n_mc = 4
        self.n_vis = 4

        self.n_input   = network_architecture['n_input']
        self.n_z  = network_architecture['n_z']

        self.x   = tf.placeholder(tf.float64, [None, self.n_input],   name='InputData')
        self.x_noiseless   = tf.placeholder(tf.float64, [None, self.n_input],   name='NoiselessData')
        
        self.layers={}

        self.n_epoch = tf.zeros([],tf.float64)

        # Create autoencoder network
        self._create_network()

        # Define loss function based variational upper-bound and corresponding optimizer
        self._create_loss_optimizer()
       
        # Initializing the tensor flow variables
        init = tf.global_variables_initializer() #tf.initialize_all_variables() # 

        # Launch the session
        self.sess = sess #tf.InteractiveSession()
        self.sess.run(init)

        self.saver = tf.train.Saver()
        
        # Summary monitors
        tf.summary.scalar("loss",self.cost) #tf.summary.FileWriter(logs_path) #
        # tf.summary.scalar("loss_J",self.cost_J)
        self.merged_summary_op = tf.summary.merge_all() #tf.merge_all_summaries()

    def _slice_input(self, input_layer, size_mod):
        slices=[]
        count =0
        for i in range(len(self.network_architecture[size_mod])):
            assert self.batch_size % self.network_architecture['size_slices'][i] == 0, \
            f"Batch size {self.batch_size} is not compatible with slice size {self.network_architecture['size_slices'][i]}"

            new_slice = tf.slice(input_layer, [0,count], [self.batch_size,self.network_architecture[size_mod][i]]) # tf.slice(layer_2, [0,200], [105,100])
            count+=self.network_architecture[size_mod][i]
            slices.append(new_slice)

        return slices

    def _create_partial_network(self,name,input_layer):
        with tf.name_scope(name):
            self.layers[name]=[input_layer]
            for i in range(len(self.network_architecture[name])):
                h=tf.Variable(xavier_init(int(self.layers[name][-1].get_shape()[1]), self.network_architecture[name][i]))
                b= tf.Variable(tf.zeros([self.network_architecture[name][i]], dtype=tf.float64))
                layer = self.transfer_fct(tf.add(tf.matmul(self.layers[name][-1],    h), b))
                self.layers[name].append(layer)
            
    def _create_variational_network(self, input_layer, latent_size):
        input_layer_size= int(input_layer.get_shape()[1])
        
        h_mean= tf.Variable(xavier_init(input_layer_size, latent_size))
        h_var= tf.Variable(xavier_init(input_layer_size, latent_size))
        b_mean= tf.Variable(tf.zeros([latent_size], dtype=tf.float64))
        b_var= tf.Variable(tf.zeros([latent_size], dtype=tf.float64))
        mean = tf.add(tf.matmul(input_layer, h_mean), b_mean)
        log_sigma_sq = tf.log(tf.exp(tf.add(tf.matmul(input_layer, h_var), b_var)) + 0.0001 )
        return mean, log_sigma_sq

    def _create_modalities_network(self, names, slices):
        for i in range(len(names)):
            self._create_partial_network(names[i],slices[i])

    def _create_mod_variational_networ(self, names, sizes_mod):
                assert len(self.network_architecture[sizes_mod])==len(names)
                sizes=self.network_architecture[sizes_mod]
                self.layers['final_means']=[]
                self.layers['final_sigmas']=[]
                for i in range(len(names)):
                        mean, log_sigma_sq=self._create_variational_network(self.layers[names[i]][-1],sizes[i])
                        self.layers['final_means'].append(mean)
                        self.layers['final_sigmas'].append(log_sigma_sq)
                global_mean=tf.concat(self.layers['final_means'],1)
                global_sigma=tf.concat(self.layers['final_sigmas'],1)
                self.layers["global_mean_reconstr"]=[global_mean]
                self.layers["global_sigma_reconstr"]=[global_sigma]
                return global_mean, global_sigma
                                       
    def _create_network(self):
        # Initialize autoencode network weights and biases
        self.x_noiseless_sliced=self._slice_input(self.x_noiseless, 'size_slices')
        slices=self._slice_input(self.x, 'size_slices')
        self._create_modalities_network(['mod0','mod1','mod2','mod3','mod4', 'mod5'], slices)

        self.output_mod = tf.concat([self.layers['mod0'][-1],self.layers['mod1'][-1],self.layers['mod2'][-1],self.layers['mod3'][-1],self.layers['mod4'][-1],self.layers['mod5'][-1]],1) 
        self.layers['concat']=[self.output_mod]
        
        #self._create_partial_network('enc_shared',self.x)
        self._create_partial_network('enc_shared',self.output_mod)
        self.z_mean, self.z_log_sigma_sq = self._create_variational_network(self.layers['enc_shared'][-1],self.n_z)

        if self.vae_mode:
                eps = tf.random_normal((self.batch_size, self.n_z), 0, 1, dtype=tf.float64)
                self.z   = tf.add(self.z_mean, tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))
        else:
                self.z   = self.z_mean
        

        self._create_partial_network('dec_shared',self.z)

        slices_shared=self._slice_input(self.layers['dec_shared'][-1], 'size_slices_shared')
        self._create_modalities_network(['mod0_2','mod1_2','mod2_2','mod3_2','mod4_2','mod5_2'], slices_shared)

        self.x_reconstr, self.x_log_sigma_sq = self._create_mod_variational_networ(['mod0_2','mod1_2','mod2_2','mod3_2','mod4_2','mod5_2'],'size_slices')
                                                                                                                     
    def _create_loss_optimizer(self):
        # The loss is composed of two terms:
        # 1.) The reconstruction loss (the negative log probability
        #     of the input under the reconstructed Bernoulli distribution 
        #     induced by the decoder in the data space).
        #     This can be interpreted as the number of "nats" required
        #     for reconstructing the input when the activation in latent
        #     is given.
        # Adding 1e-10 to avoid evaluation of log(0.0)
        # 2.) The latent loss, which is defined as the Kullback Leibler divergence 
        ##    between the distribution in latent space induced by the encoder on 
        #     the data and some prior. This acts as a kind of regularizer.
        #     This can be interpreted as the number of "nats" required
        #     for transmitting the the latent space distribution given
        #     the prior.
        with tf.name_scope('Loss_Opt'):
                        self.alpha = 1- tf.minimum(self.n_epoch/1000, 1) # the coefficent used to reduce the impact of latent loss

                        self.tmp_costs=[]
                        for i in range(len(self.layers['final_means'])):
                                reconstr_loss = ( 0.5 * tf.reduce_sum(tf.square(self.x_noiseless_sliced[i] - self.layers['final_means'][i]) / tf.exp(self.layers['final_sigmas'][i]),1) \
                                                + 0.5 * tf.reduce_sum(self.layers['final_sigmas'][i],1) \
                                                + 0.5 * self.n_z/2 * np.log(2*math.pi) )/self.network_architecture['size_slices'][i]
                                self.tmp_costs.append(reconstr_loss)
                                
                        self.reconstr_loss = tf.reduce_mean(self.tmp_costs[0]+ self.tmp_costs[1] + self.tmp_costs[2] + self.tmp_costs[3] + self.tmp_costs[4] + self.tmp_costs[5])

                        self.latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq  - tf.square(self.z_mean)  - tf.exp(self.z_log_sigma_sq), 1)

                        self.cost = tf.reduce_mean(self.reconstr_loss + tf.scalar_mul( self.alpha, self.latent_loss))  # average over batch

                        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)  # Use ADAM optimizer

                        self.m_reconstr_loss = self.reconstr_loss
                        self.m_latent_loss = tf.reduce_mean(self.latent_loss)         

    def print_layers_size(self):
        print(self.cost)
        for layer in self.layers:
            print(layer)
            for l in self.layers[layer]:
                print(l)

    def partial_fit(self,sess, X, X_noiseless, epoch):
        """Train model based on mini-batch of input data.
        
        Return cost of mini-batch.
        """

        opt, cost, recon, latent, x_rec, alpha = sess.run((self.optimizer, self.cost, self.m_reconstr_loss,self.m_latent_loss, self.x_reconstr, self.alpha), 
            feed_dict={self.x: X, self.x_noiseless: X_noiseless, self.n_epoch: epoch})
        return cost, recon, latent, x_rec, alpha

    def transform(self,sess, X):
        """Transform data by mapping it into the latent space."""
        # Note: This maps to mean of distribution, we could alternatively sample from Gaussian distribution
        return sess.run(self.z_mean, feed_dict={self.x: X})
    
    def generate(self,sess, z_mu=None):
        """ Generate data by sampling from latent space.
        
        If z_mu is not None, data for this point in latent space is generated. Otherwise, z_mu is drawn from prior in latent space.        
        """
        if z_mu is None:
            z_mu = np.random.normal(size=self.n_z)
        # Note: This maps to mean of distribution, we could alternatively sample from Gaussian distribution
        return sess.run(self.x_reconstr, feed_dict={self.z: z_mu})
    
    def reconstruct(self,sess, X_test):
        """ Use VAE to reconstruct given data. """
        x_rec_mean,x_rec_log_sigma_sq = sess.run((self.x_reconstr, self.x_log_sigma_sq), 
            feed_dict={self.x: X_test})
        return x_rec_mean,x_rec_log_sigma_sq
def xavier_init(fan_in, fan_out, constant=1): 
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(1.0/(fan_in + fan_out)) 
    high = constant*np.sqrt(1.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),  minval=low, maxval=high,  dtype=tf.float64)

