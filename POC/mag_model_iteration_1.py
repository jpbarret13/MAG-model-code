#!/usr/bin/env python
# coding: utf-8

# In[1]:


# %pip install tensorflow==2.4.1
# %pip install transformers
# %pip install pyarrow
# %pip install tensorflow-addons


# In[1]:


import tensorflow as tf
import pandas as pd
import pickle
import os
import tensorflow_addons as tfa
from transformers import RobertaTokenizer, RobertaTokenizerFast, TFRobertaModel, TFAlbertModel

AUTO = tf.data.experimental.AUTOTUNE


# In[2]:


model_iteration = 'iteration_1'


# In[3]:


tf.config.list_physical_devices()


# In[4]:


with open(f"./{model_iteration}/vocab/topics_vocab.pkl", "rb") as f:
    target_vocab = pickle.load(f)
    
with open(f"./{model_iteration}/vocab/doc_type_vocab.pkl", "rb") as f:
    doc_vocab = pickle.load(f)
    
with open(f"./{model_iteration}/vocab/journal_name_vocab.pkl", "rb") as f:
    journal_vocab = pickle.load(f)


# In[5]:


encoding_layer = tf.keras.layers.experimental.preprocessing.CategoryEncoding(
    max_tokens=len(target_vocab)+1, output_mode="binary", sparse=False)

# loss_fn = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
loss_fn = tfa.losses.SigmoidFocalCrossEntropy(alpha=0.25, gamma=2.0, 
                                              reduction=tf.keras.losses.Reduction.NONE)

metric_1 = tf.keras.metrics.CategoricalAccuracy()
metric_2 = tf.keras.metrics.Recall()
metric_3 = tf.keras.metrics.Precision()
metric_4 = tf.keras.metrics.TopKCategoricalAccuracy(k=10)
# Eventually will use with focal loss


# In[6]:


class CustomModel(tf.keras.Model):
    def train_step(self, inputs):
        old_features, labels = inputs
        labels = tf.RaggedTensor.from_tensor(labels, padding=0)
        paper_titles = old_features[0][:,:512].to_tensor(shape=[None, 512])
        paper_masks = old_features[1][:,:512].to_tensor(shape=[None, 512])
        
        features = (paper_titles, paper_masks, old_features[2], old_features[3])
        labels = encoding_layer(labels)

        with tf.GradientTape() as tape:
            predictions = self(features, training=True)
            loss = loss_fn(labels, predictions)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        metric_1.update_state(labels, predictions)
        metric_2.update_state(labels, predictions)
        metric_3.update_state(labels, predictions)
        metric_4.update_state(labels, predictions)

        return {"loss": loss, 
                "accuracy": metric_1.result(), 
                "recall": metric_2.result(), 
                "precision": metric_3.result(), 
                "topK15": metric_4.result()}
  
    def test_step(self, inputs):
        old_features, labels = inputs
        labels = tf.RaggedTensor.from_tensor(labels, padding=0)
        paper_titles = old_features[0][:,:512].to_tensor(shape=[None, 512])
        paper_masks = old_features[1][:,:512].to_tensor(shape=[None, 512])
        
        features = (paper_titles, paper_masks, old_features[2], old_features[3])
        labels = encoding_layer(labels)

        with tf.GradientTape() as tape:
            predictions = self(features, training=False)
            loss = loss_fn(labels, predictions)

        metric_1.update_state(labels, predictions)
        metric_2.update_state(labels, predictions)
        metric_3.update_state(labels, predictions)
        metric_4.update_state(labels, predictions)

        return {"loss": loss, 
                "accuracy": metric_1.result(), 
                "recall": metric_2.result(), 
                "precision": metric_3.result(), 
                "topK15": metric_4.result()}
    @property
    def metrics(self):
        return [metric_1, metric_2, metric_3]


# In[7]:


def _parse_function(example_proto):

    feature_description = {
        'paper_title': tf.io.RaggedFeature(tf.int64),
        'paper_mask': tf.io.RaggedFeature(tf.int64),
        'journal': tf.io.FixedLenFeature((1,), tf.int64),
        'doc_type': tf.io.FixedLenFeature((1,), tf.int64),
        'targets': tf.io.FixedLenFeature((20,), tf.int64)
    }

    example = tf.io.parse_single_example(example_proto, feature_description)

    paper_title = example['paper_title']
    paper_mask = example['paper_mask']
    doc_type = example['doc_type']
    journal = example['journal']
    targets = example['targets']

    return (paper_title, paper_mask, doc_type, journal), targets


# In[8]:


def get_dataset(path, data_type='train'):
    
    tfrecords = [f"{path}{data_type}/{x}" for x in os.listdir(f"{path}{data_type}/") if x.endswith('tfrecord')]
    tfrecords.sort()
    
    
    raw_dataset = tf.data.TFRecordDataset(tfrecords[:25], num_parallel_reads=AUTO)

    parsed_dataset = raw_dataset.map(_parse_function, num_parallel_calls=AUTO)

    parsed_dataset = parsed_dataset     .apply(tf.data.experimental.dense_to_ragged_batch(256, drop_remainder=True)).shuffle(1024)
    return parsed_dataset.prefetch(AUTO)


# In[9]:


file_path = f'./{model_iteration}/tfrecords/'


# In[10]:


train_ds = get_dataset(file_path, 'train')
val_ds = get_dataset(file_path, 'val')


# In[11]:


mirrored_strategy = tf.distribute.MirroredStrategy()


# In[12]:


with mirrored_strategy.scope():
#     model = TFAlbertModel.from_pretrained('albert-base-v2')
#     model.layers[0].trainable = False

    # Model Inputs
    paper_title_input_ids = tf.keras.layers.Input((512,), dtype=tf.int64, name='paper_title_ids')
    paper_title_att_mask = tf.keras.layers.Input((512,), dtype=tf.int64, name='paper_title_mask')
    doc_type_id = tf.keras.layers.Input((1,), dtype=tf.int64, name='doc_type_id')
    journal_id = tf.keras.layers.Input((1,), dtype=tf.int64, name='journal_id')

    # Using HF Model for Title Representation
#     paper_title_embs = model(input_ids = paper_title_input_ids, 
#                              attention_mask=paper_title_att_mask, 
#                              output_hidden_states=True, 
#                              training=False).last_hidden_state

    # Embedding Layers
    paper_title_embs = tf.keras.layers.Embedding(input_dim=30001, 
                                                 output_dim=512, 
                                                 mask_zero=False, 
                                                 trainable=True,
                                                 name="title_embedding")(paper_title_input_ids)

    doc_embs = tf.keras.layers.Embedding(input_dim=len(doc_vocab)+1, 
                                          output_dim=32, 
                                          mask_zero=False, 
                                          name="doc_type_embedding")(doc_type_id)

    journal_embs = tf.keras.layers.Embedding(input_dim=len(journal_vocab)+1, 
                                              output_dim=128, 
                                              mask_zero=False, 
                                              name="journal_embedding")(journal_id)

    # First layer
    dense_output = tf.keras.layers.Dense(1024, activation='relu', 
                                         kernel_regularizer='L2', name="dense_1")(paper_title_embs)
    dense_output = tf.keras.layers.Dropout(0.20, name="dropout_1")(dense_output)
    dense_output = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="layer_norm_1")(dense_output)
    dense_output_flat = tf.keras.layers.GlobalAveragePooling1D(name="title_pooling_layer")(dense_output)
    doc_flat = tf.keras.layers.GlobalAveragePooling1D(name="doc_pooling_layer")(doc_embs)
    journal_flat = tf.keras.layers.GlobalAveragePooling1D(name="journal_pooling_layer")(journal_embs)
    concat_output = tf.concat(values=[dense_output_flat, journal_flat, doc_flat], axis=1)

    # Second layer
    dense_output = tf.keras.layers.Dense(1024, activation='relu', 
                                         kernel_regularizer='L2', name="dense_2")(concat_output)
    dense_output = tf.keras.layers.Dropout(0.20, name="dropout_2")(dense_output)
    dense_output = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="layer_norm_2")(dense_output)

    # Third Layer
    dense_output = tf.keras.layers.Dense(256, activation='relu', 
                                         kernel_regularizer='L2', name="dense_3")(dense_output)
    dense_output = tf.keras.layers.Dropout(0.20, name="dropout_3")(dense_output)
    dense_output = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="layer_norm_3")(dense_output)
#     dense_output_flat = tf.keras.layers.GlobalAveragePooling1D(name="title_pooling_layer")(dense_output)


    # Output Layer
    final_output = tf.keras.layers.Dense(len(target_vocab)+1, activation="sigmoid", 
                                         name="cls")(dense_output)

    test_model = CustomModel(inputs=[paper_title_input_ids, paper_title_att_mask, doc_type_id, journal_id], 
                             outputs=final_output, name='test_model')

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


# In[13]:


test_model.compile(optimizer=optimizer)


# In[14]:


test_model.summary()


# In[15]:


callbacks = [tf.keras.callbacks.ModelCheckpoint(f'./models/{model_iteration}/{model_iteration}_first_try', 
                                                save_best_only=False, save_weights_only=False)]


# ## First try (with all variables and Albert model output)

# In[ ]:


history = test_model.fit(train_ds, epochs=1, validation_data=val_ds, verbose=1, callbacks=callbacks)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## ARCHIVE: Baseline Second Try (trainable embeddings)

# In[23]:


history = test_model.fit(train_ds, epochs=5, validation_data=val_ds, verbose=1, callbacks=callbacks)


# In[ ]:





# In[ ]:





# In[ ]:




