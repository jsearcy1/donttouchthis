import tensorflow as tf



application=tf.keras.applications.mobilenet.MobileNet(input_shape=(96,96,3),include_top=False,weights='imagenet')

for layers in application.layers:
    layers.trainable=False


head=tf.keras.layers.Flatten()(application.output)
head=tf.keras.layers.Dropout(.3)(head)

head=tf.keras.layers.Dense(128)(head)
head=tf.keras.layers.GaussianNoise(.1)(head)
head=tf.keras.layers.LeakyReLU()(head)
head=tf.keras.layers.Dropout(.3)(head)

features=tf.keras.layers.Dense(128)(head)
head=tf.keras.layers.GaussianNoise(.1)(features)
head=tf.keras.layers.LeakyReLU()(head)

head=tf.keras.layers.Dropout(.3)(head)
head=tf.keras.layers.Dense(1,activation='sigmoid')(head)
    

model=tf.keras.models.Model(application.input,[head])
adam=tf.keras.optimizers.Adam(lr=1e-4)
model.compile(loss=['binary_crossentropy'],optimizer=adam,metrics=['accuracy'])

preprocess=tf.keras.applications.mobilenet.preprocess_input
