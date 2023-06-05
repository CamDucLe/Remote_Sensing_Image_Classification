import gradio as gr
import tensorflow as tf
import numpy as np
import json

class Contrastive_Loss_2(tf.keras.losses.Loss):
  def __init__(self, temperature=0.5, rate=0.5, name='Contrastive_Loss_2', **kwargs):
    super(Contrastive_Loss_2, self).__init__(name=name, **kwargs)
    self.temperature   = temperature
    self.rate          = rate
    self.cosine_sim    = tf.keras.losses.CosineSimilarity(axis=-1, reduction=tf.keras.losses.Reduction.NONE)
    
  # @tf.function
  def call(self, z1, z2):
    batch_size, n_dim = z1.shape

    # Compute Euclid Distance loss
    difference    = z1 - z2                                             # (BxB)   * z1 and z2 already applied soft max -> in the last axis, max dif will be 1 
    squared_norm  = tf.reduce_sum(tf.square(difference), axis=1)        # (B)
    distance      = tf.sqrt(squared_norm + 1e-8)                        # (B)     * + epsilon to avoid Nan in gradient
    mean_distance = tf.reduce_mean(distance)                            # () -> scalar
    tf.debugging.check_numerics(mean_distance.numpy(), 'Distance contains NaN values.')
    # print('distance: , ',mean_distance)

    # Compute Consine Similarity loss
    z = tf.concat((z1, z2), 0)

    sim_ij      = - self.cosine_sim(z[:batch_size], z[batch_size:])     # (B)  -> batch_size pair
    sim_ji      = - self.cosine_sim(z[batch_size:], z[:batch_size])     # (B)  -> batch_size pair
    sim_pos     = tf.concat((sim_ij,sim_ji), axis=0)                    # (2B) -> 2*batch_size positive pair
    numerator   = tf.math.exp(sim_pos / self.temperature)               # (2B) -> 2*batch_size positive pair
  
    sim_neg     = - self.cosine_sim(tf.expand_dims(z, 1), z)            # sim (Bx1xE, BxE) -> (2Bx2B)
    mask        = 1 - tf.eye(2*batch_size, dtype=tf.float32)            # (2Bx2B)
    sim_neg     = mask * tf.math.exp(sim_neg / self.temperature)        # (2Bx2B)
    denominator = tf.math.reduce_sum(sim_neg, axis=-1)                  # (2B) 
  
    mean_cosine_similarity = tf.reduce_mean(- tf.math.log((numerator + 1e-11) / (denominator + 1e-11)))       # () -> scalar
    tf.debugging.check_numerics(mean_cosine_similarity.numpy(), 'Cosine contains NaN values.')
    # print('similarity: , ',mean_cosine_similarity)

    # Compute total loss with associated rate
    total_loss = (1-self.rate)*mean_distance + self.rate*mean_cosine_similarity 
    tf.debugging.check_numerics(total_loss.numpy(), 'Total contains NaN values.')
    return total_loss
    
model = tf.keras.models.load_model( filepath='contrastive_model.h5',  custom_objects={'Contrastive_Loss_2': Contrastive_Loss_2}, compile=False)

with open("scene_labels.json") as labels_file:
   labels = json.load(labels_file)

def classify_image(img):
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = tf.expand_dims(img, axis=0)
    probs = (model(img))[0]
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

image = gr.inputs.Image(shape=(256, 256))
label = gr.outputs.Label(num_top_classes=5) 
examples = ['images/right/roundabout_028.jpg',
            'images/right/airplane_002.jpg',
            'images/right/baseball_diamond_018.jpg',
            'images/right/meadow_019.jpg',
            'images/right/ship_002.jpg',
            'images/right/storage_tank_002.jpg',
            'images/right/freeway_002.jpg',
            'images/right/overpass_015.jpg',
            'images/right/airport_002.jpg',
            'images/right/beach_002.jpg',
            'images/wrong/airport_020.jpg',
            'images/wrong/palace_004.jpg',
            'images/wrong/desert_199.jpg',
            ]

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
intf.launch(inline=False)

