import tensorflow as tf
from config import *

dense_initializer = tf.keras.initializers.RandomNormal()

weights = {
    "gan_gen_w0" : tf.Variable(dense_initializer([GEN_SEED_SIZE, GEN_L0_SIZE])),
    "gan_gen_b0" : tf.Variable(dense_initializer([GEN_L0_SIZE])),

    "gan_gen_w1" : tf.Variable(dense_initializer([GEN_L0_SIZE, GEN_L1_SIZE])),
    "gan_gen_b1" : tf.Variable(dense_initializer([GEN_L1_SIZE])),

    "gan_gen_w2" : tf.Variable(dense_initializer([GEN_L1_SIZE, GEN_OUTPUT_SIZE])),
    "gan_gen_b2" : tf.Variable(dense_initializer([GEN_OUTPUT_SIZE])),


    "dgan_gen_w0" : tf.Variable(dense_initializer([GEN_SEED_SIZE, GEN_L0_SIZE])),
    "dgan_gen_b0" : tf.Variable(dense_initializer([GEN_L0_SIZE])),

    "dgan_gen_w1" : tf.Variable(dense_initializer([GEN_L0_SIZE, GEN_L1_SIZE])),
    "dgan_gen_b1" : tf.Variable(dense_initializer([GEN_L1_SIZE])),

    "dgan_gen_w2" : tf.Variable(dense_initializer([GEN_L1_SIZE, GEN_OUTPUT_SIZE])),
    "dgan_gen_b2" : tf.Variable(dense_initializer([GEN_OUTPUT_SIZE])),


    "gan_disc_w0" : tf.Variable(dense_initializer([GEN_OUTPUT_SIZE, DISC_L0_SIZE])),
    "gan_disc_b0" : tf.Variable(dense_initializer([DISC_L0_SIZE])),

    "gan_disc_w1" : tf.Variable(dense_initializer([DISC_L0_SIZE, DISC_L1_SIZE])),
    "gan_disc_b1" : tf.Variable(dense_initializer([DISC_L1_SIZE])),

    "gan_disc_w2" : tf.Variable(dense_initializer([DISC_L1_SIZE, 1])),
    "gan_disc_b2" : tf.Variable(dense_initializer([1])),


    "dgan_disc_w0" : tf.Variable(dense_initializer([DGAN_COMPARE_ENTRIES * GEN_OUTPUT_SIZE, DGAN_L0_SIZE])),
    "dgan_disc_b0" : tf.Variable(dense_initializer([DGAN_L0_SIZE])),

    "dgan_disc_w1" : tf.Variable(dense_initializer([DGAN_L0_SIZE, DGAN_L1_SIZE])),
    "dgan_disc_b1" : tf.Variable(dense_initializer([DGAN_L1_SIZE])),

    "dgan_disc_w2" : tf.Variable(dense_initializer([DGAN_L1_SIZE, 1])),
    "dgan_disc_b2" : tf.Variable(dense_initializer([1]))
}

def dense(input, w, b, activation, dropout):
    #right multiply for proper broadcasting operations
    x = tf.matmul(input, weights[w]);
    x = tf.add(weights[b], x);
    x = activation(x);

    if dropout: 
        x = tf.nn.dropout(x, DROPOUT);

    return x;

def gan_gen_nn(random_seed, training):
    l0 = dense(random_seed,"gan_gen_w0", "gan_gen_b0", tf.nn.leaky_relu, training); 
    l1 = dense(l0, "gan_gen_w1", "gan_gen_b1", tf.nn.leaky_relu, training); 
    l2 = dense(l1, "gan_gen_w2", "gan_gen_b2", tf.nn.sigmoid, False);

    return l2; 

def dgan_gen_nn(random_seed, training):
    l0 = dense(random_seed,"dgan_gen_w0", "dgan_gen_b0", tf.nn.leaky_relu, training); 
    l1 = dense(l0, "dgan_gen_w1", "dgan_gen_b1", tf.nn.leaky_relu, training); 
    l2 = dense(l1, "dgan_gen_w2", "dgan_gen_b2", tf.nn.sigmoid, False);

    return l2; 

#an output of 1 indicates real
def gan_disc_nn(gen_output, training):
    l0 = dense(gen_output,"gan_disc_w0", "gan_disc_b0", tf.nn.leaky_relu, training); 
    l1 = dense(l0, "gan_disc_w1", "gan_disc_b1", tf.nn.leaky_relu, training); 
    l2 = dense(l1, "gan_disc_w2", "gan_disc_b2", tf.nn.sigmoid, False);

    return l2;

def dgan_disc_nn(gen_outputs, training):
    l0 = dense(gen_outputs,"dgan_disc_w0", "dgan_disc_b0", tf.nn.leaky_relu, training); 
    l1 = dense(l0, "dgan_disc_w1", "dgan_disc_b1", tf.nn.leaky_relu, training); 
    l2 = dense(l1, "dgan_disc_w2", "dgan_disc_b2", tf.nn.sigmoid, False);

    return l2;

bce = tf.keras.losses.BinaryCrossentropy();
#loss
def gen_loss(disc_prediction):
    return bce(tf.ones_like(disc_prediction), disc_prediction)

def disc_loss(disc_prediction_gen, disc_prediction_real):
    return bce(tf.zeros_like(disc_prediction_gen), disc_prediction_gen) + bce(tf.ones_like(disc_prediction_real), disc_prediction_real); 

gan_gen_optimizer = tf.keras.optimizers.Adam(GEN_LEARNING_RATE);
gan_disc_optimizer = tf.keras.optimizers.Adam(DISC_LEARNING_RATE); 
dgan_gen_optimizer = tf.keras.optimizers.Adam(GEN_LEARNING_RATE);
dgan_disc_optimizer = tf.keras.optimizers.Adam(DISC_LEARNING_RATE); 

#training data is just [None, 2]
def dgan_train(training_data):
    if (len(training_data) % (DGAN_BATCH_SIZE * DGAN_COMPARE_ENTRIES)):
        print("DGAN: bad training data length");   
        return
 
    gen_weights = [weights["dgan_gen_w0"], weights["dgan_gen_b0"], weights["dgan_gen_w1"], weights["dgan_gen_b1"], weights["dgan_gen_w2"], weights["dgan_gen_b2"]] 
    disc_weights = [weights["dgan_disc_w0"], weights["dgan_disc_b0"], weights["dgan_disc_w1"], weights["dgan_disc_b1"], weights["dgan_disc_w2"], weights["dgan_disc_b2"]] 
    random_seed = tf.random.normal([DGAN_COMPARE_ENTRIES, DGAN_BATCH_SIZE, GEN_SEED_SIZE], stddev = 1)
    
    for i in range(EPOCHS):
        for j in range(len(training_data) // (DGAN_BATCH_SIZE)):
            batch = training_data[j * DGAN_BATCH_SIZE : (j + 1) * DGAN_BATCH_SIZE];
            batch = tf.reshape(batch, [-1, DGAN_COMPARE_ENTRIES * GEN_OUTPUT_SIZE])
            
            #split into batches
            #delineate batches into subbtaches

            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                #create a run of gen res and add together
                
                gen_res = tf.concat([dgan_gen_nn(random_seed[x], True) for x in range(DGAN_COMPARE_ENTRIES)], axis = 1)
                disc = dgan_disc_nn(gen_res, True);
                disc_real = dgan_disc_nn(batch, True);        
         
                gloss = gen_loss(disc);
                dloss = disc_loss(disc, disc_real);

            dg = gen_tape.gradient(gloss, gen_weights); 
            dd = disc_tape.gradient(dloss, disc_weights);
            
            #train gan and disc
            dgan_gen_optimizer.apply_gradients(zip(dg, gen_weights))
            dgan_disc_optimizer.apply_gradients(zip(dd, disc_weights))
            
            if (j == len(training_data) // DGAN_BATCH_SIZE - 1): 
                print("DGAN: Epoch:",i,"gloss", gloss.numpy(), "dloss", dloss.numpy()); 

        yield dgan_gen_nn(random_seed[0], True), dgan_disc_nn; 
 
def gan_train(training_data):
    gen_weights = [weights["gan_gen_w0"], weights["gan_gen_b0"], weights["gan_gen_w1"], weights["gan_gen_b1"], weights["gan_gen_w2"], weights["gan_gen_b2"]] 
    disc_weights = [weights["gan_disc_w0"], weights["gan_disc_b0"], weights["gan_disc_w1"], weights["gan_disc_b1"], weights["gan_disc_w2"], weights["gan_disc_b2"]] 
    random_seed = tf.random.normal([GAN_BATCH_SIZE, GEN_SEED_SIZE], stddev = 1)

    for i in range(EPOCHS):
        for j in range(len(training_data) // (GAN_BATCH_SIZE)):
            batch = training_data[j * GAN_BATCH_SIZE : (j + 1) * GAN_BATCH_SIZE];
            
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                gen_res = gan_gen_nn(random_seed, True) 
                disc = gan_disc_nn(gen_res, True);
                disc_real = gan_disc_nn(batch, True);        
         
                gloss = gen_loss(disc);
                dloss = disc_loss(disc, disc_real);
                           

            dg = gen_tape.gradient(gloss, gen_weights); 
            dd = disc_tape.gradient(dloss, disc_weights);
            
            #train gan and disc
            gan_gen_optimizer.apply_gradients(zip(dg, gen_weights))
            gan_disc_optimizer.apply_gradients(zip(dd, disc_weights))

            if (j == len(training_data) // GAN_BATCH_SIZE - 1): 
                print("GAN:  Epoch:",i,"gloss", gloss.numpy(), "dloss", dloss.numpy());
        yield gen_res, gan_disc_nn; 
