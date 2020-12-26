import os
import numpy as np
import cv2
from glob import glob
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow_examples.models.pix2pix import pix2pix
from IPython.display import clear_output
import pathlib
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import math
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
import argparse

batch_size = 8
img_height = 256
img_width = 256

#Set path to the data 
train_path = "./train2/"

# Generate full path to from the dataset path on the PC
train_dir = pathlib.Path(train_path)


# Plot images
def plot_images(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def generate_images():
  generators = [
    ImageDataGenerator(rotation_range=135), 
    ImageDataGenerator(width_shift_range=.15), 
    ImageDataGenerator(height_shift_range=.15),
    ImageDataGenerator(zoom_range=0.5),
    ImageDataGenerator(vertical_flip=True),
    ImageDataGenerator(horizontal_flip=True)
    ]

  prefix = ['rt',"ws", "hs", "zr", "vf", "hf"]

  for i in range(len(generators)):
    num_itr = math.ceil(74/batch_size)
    seed = 42
    img_gen = generators[i].flow_from_directory(
        'train2/input_image',
        class_mode=None,
        batch_size = batch_size,
        save_to_dir = "train2/image_gen",
        save_prefix= prefix[i],
        seed=seed)

    for img_batch in range (num_itr):
      img_gen.next()
    mask_gen = generators[i].flow_from_directory(
        'train2/segmentation_true',
        class_mode=None,
        batch_size = batch_size,
        save_to_dir = "train2/mask_gen",
        save_prefix= prefix[i],
        seed=seed)
    for mask_batch in range (num_itr):
      mask_gen.next()



# Load dataset from directory in the folder
def load_data(path, split=0.1):
  images = sorted(glob(os.path.join(path, "input_image/images/*")))
  masks = sorted(glob(os.path.join(path, "segmentation_true/images/*")))

  total_size = len(images)
  valid_size = int(split * total_size)
  test_size = int(split * total_size)

  train_x, valid_x = train_test_split(images, test_size=valid_size, random_state=42)
  train_y, valid_y = train_test_split(masks, test_size=valid_size, random_state=42)

  train_x, test_x = train_test_split(train_x, test_size=test_size, random_state=42)
  train_y, test_y = train_test_split(train_y, test_size=test_size, random_state=42)

  return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)



def read_image(path):
  path = path.decode()
  x = cv2.imread(path, cv2.IMREAD_COLOR)
  x = cv2.resize(x, (img_width, img_height))
  x = x/255.0
  return x


def read_mask(path):
  path = path.decode()
  x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
  x = cv2.resize(x, (img_width, img_height))
  x = x/255.0
  x = np.expand_dims(x, axis=-1)
  return x


def tf_parse(x, y):
  def _parse(x, y):
      x = read_image(x)
      y = read_mask(y)
      return x, y

  x, y = tf.numpy_function(_parse, [x, y], [tf.float64, tf.float64])
  x.set_shape([img_width, img_height, 3])
  y.set_shape([img_width, img_height, 1])


  return x, y


def tf_dataset(x, y, batch=8):
  dataset = tf.data.Dataset.from_tensor_slices((x, y))
  dataset = dataset.map(tf_parse)
  return dataset



def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()



(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(train_path)


train = tf_dataset(train_x, train_y)
test = tf_dataset(test_x, test_y)
val = tf_dataset(valid_x, valid_y)



for image, mask in train.take(1):
  sample_image, sample_mask = image, mask

TRAIN_LENGTH = len(train)
BATCH_SIZE = batch_size
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

# Configure for performance
train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
validation_dataset = val.batch(BATCH_SIZE)
test_dataset = val.batch(BATCH_SIZE)

# Specifying output channel
OUTPUT_CHANNELS = 3

#Use MobileNetV2 as base model and for transfer learning
base_model = tf.keras.applications.MobileNetV2(input_shape=[img_width, img_height, 3], include_top=False)

# Use the activations of these layers
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]
layers = [base_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

down_stack.trainable = False

up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]


def unet_model(output_channels):
  inputs = tf.keras.layers.Input(shape=[img_width, img_height, 3])
  x = inputs

  # Downsampling through the model
  skips = down_stack(x)
  x = skips[-1]
  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    concat = tf.keras.layers.Concatenate()
    x = concat([x, skip])

  # This is the last layer of the model
  last = tf.keras.layers.Conv2DTranspose(
      output_channels, 3, strides=2,
      padding='same')  #64x64 -> 256x256

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)



model = unet_model(OUTPUT_CHANNELS)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# tf.keras.utils.plot_model(model, show_shapes=True) # Print model structure


def create_mask(pred_mask,img_bn):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[img_bn]


def show_predictions(model, dataset=None, num=1):
  if dataset:
    for image, mask in dataset.take(num):
      print(image.shape)
      pred_mask = model.predict(image)
      display([image[0], mask[0], create_mask(pred_mask,0)])
  else:
    display([sample_image, sample_mask,
             create_mask(model.predict(sample_image[tf.newaxis, ...]),0)])


EPOCHS = 40
VAL_SUBSPLITS = 5
VALIDATION_STEPS = len(val)//BATCH_SIZE//VAL_SUBSPLITS


#Call this function if you want to retrain from scratch
def train_model():
  callbacks = [
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4),
          #  DisplayCallback(),
            TensorBoard(),
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False)
      ]
  model.fit(train_dataset, epochs=EPOCHS,
                            steps_per_epoch=STEPS_PER_EPOCH,
                            validation_steps=VALIDATION_STEPS,
                            validation_data=validation_dataset,
                            callbacks=callbacks)

  #This function saves the model
  model.save("my_model")

# Load saved model
reconstructed_model = tf.keras.models.load_model("my_model")


def predict_and_save():
  ap = argparse.ArgumentParser()
  ap.add_argument("-i", "--image_folder", type=str, help="path to input image")
  ap.add_argument("-of", "--output_folder", type=str, help="path to output folder") 
  ap.add_argument("-n", "--num", type=bool, default=False, help="Whether run it everything or just one")
  args = vars(ap.parse_args())

  if args["num"]:
    filenames = os.listdir(os.path.join("test",args["image_folder"]))
    images = sorted(glob(os.path.join("test",args["image_folder"]+"/*")))
    prediction_dataset = tf_dataset(images, images).batch(BATCH_SIZE)
    j = 0
    
    img_len = len(images)
    for image_batch,image2 in prediction_dataset.take(math.ceil(img_len/batch_size)):
      print(image_batch.shape)
      
      for bn in range(image_batch.shape[0]):
        pred_mask = create_mask(reconstructed_model.predict(image_batch),bn)

        #Load original image and get it's size to resize the preduicted mask
        original_img = cv2.imread(images[j])
        h, w, _ = original_img.shape

        pred_mask = tf.keras.preprocessing.image.array_to_img(pred_mask).resize(size=(w, h))
        # print(type(pred_mask))
        pred_mask = cv2.cvtColor(np.array(pred_mask), cv2.COLOR_RGB2BGR)
        # pred_mask = cv2.cvtColor(pred_mask, cv2.COLOR_BGR2GRAY)
        # Otsu's thresholding
        # ret2,th2 = cv2.threshold(pred_mask,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        plt.imsave(os.path.join(args["output_folder"],filenames[j]+".out.jpg"),pred_mask)
        j=j+1
  else:
    image = [args["image_folder"]]
    prediction_dataset = tf_dataset(image, image).batch(BATCH_SIZE)
    for image_batch,image2 in prediction_dataset.take(1):
      pred_mask = create_mask(reconstructed_model.predict(image_batch),0)
       #Load original image and get it's size to resize the preduicted mask
      original_img = cv2.imread(image[0])
      h, w, _ = original_img.shape

      pred_mask = tf.keras.preprocessing.image.array_to_img(pred_mask).resize(size=(w, h))
      # print(type(pred_mask))
      pred_mask = cv2.cvtColor(np.array(pred_mask), cv2.COLOR_RGB2BGR)
      # pred_mask = cv2.cvtColor(pred_mask, cv2.COLOR_BGR2GRAY)
      # Otsu's thresholding
      # ret2,th2 = cv2.threshold(pred_mask,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

      plt.imsave(os.path.join(args["output_folder"],image[0]+".out.jpg"),pred_mask)



predict_and_save()