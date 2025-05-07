import os
import json  
import keras_tuner as kt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNet
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

path = '/Users/ardhyantry/Documents/COMVI/cat breeds1'

img_size = 224

train_datagen = ImageDataGenerator(
    rescale=1./255,                     
    rotation_range=30,                  
    width_shift_range=0.2,              
    height_shift_range=0.2,             
    shear_range=0.2,                    
    zoom_range=0.2,                     
    horizontal_flip=True,               
    fill_mode='nearest',                
    validation_split=0.2,               
    brightness_range=[0.2, 1.0],        
    channel_shift_range=50.0            
)

train_generator = train_datagen.flow_from_directory(
    path,                                
    target_size=(img_size, img_size),    
    batch_size=32,                       
    class_mode='categorical',            
    subset='training'                    
)

validation_generator = train_datagen.flow_from_directory(
    path,
    target_size=(img_size, img_size),
    batch_size=32,
    class_mode='categorical',
    subset='validation'                 
)

class_indices = train_generator.class_indices
with open('class_indices.json', 'w') as f:
    json.dump(class_indices, f)
    print("class_indices telah disimpan ke 'class_indices.json'")

def build_model(hp):
    base_model = MobileNet(weights='imagenet', input_shape=(img_size, img_size, 3), include_top=False)

    for layer in base_model.layers[:hp.Int('frozen_layers', min_value=50, max_value=85, step=5)]:
        layer.trainable = False

    x = layers.GlobalAveragePooling2D()(base_model.output)  
    x = layers.Dense(
        hp.Int('dense_units', min_value=128, max_value=512, step=128),
        activation='relu',
        kernel_regularizer=regularizers.l2(0.001)
    )(x)  
    x = layers.Dropout(hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1))(x) 
    x = layers.BatchNormalization()(x)  
    x = layers.Dense(train_generator.num_classes, activation='softmax')(x)  

    model = Model(inputs=base_model.input, outputs=x)

    model.compile(optimizer=Adam(learning_rate=hp.Float('learning_rate', min_value=1e-6, max_value=1e-3, sampling='LOG')),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    return model

tuner = kt.RandomSearch(
    build_model,  
    objective='val_accuracy', 
    max_trials=20,  
    executions_per_trial=1, 
    directory='my_dir', 
    project_name='cat_breed_tuning_mobilenet' 
)


lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)


early_stopping = EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1)
model_checkpoint = ModelCheckpoint('cat_breed_classifier_best_mobilenet.keras',
                                   monitor='val_loss',
                                   save_best_only=True,
                                   verbose=1)


tuner.search(train_generator, epochs=20, validation_data=validation_generator, callbacks=[early_stopping, model_checkpoint])


best_model = tuner.get_best_models(num_models=1)[0]
best_hp = tuner.oracle.get_best_trials(num_trials=1)[0].hyperparameters


print("Best Hyperparameters: ", best_hp.values)


for layer in best_model.layers:
    layer.trainable = True  

best_model.compile(optimizer=Adam(learning_rate=best_hp.values['learning_rate']),
                   loss='categorical_crossentropy', metrics=['accuracy'])

history = best_model.fit(
    train_generator,
    epochs=20,  # Train for more epochs after tuning
    validation_data=validation_generator,
    verbose=1,
    callbacks=[early_stopping, model_checkpoint, lr_scheduler]
)

# Evaluate the model
test_loss, test_acc = best_model.evaluate(validation_generator)
print(f"Test Accuracy: {test_acc*100:.2f}%")

# Save the trained model
best_model.save('cat_breed_classifier_final_mobilenet.keras')
print("Model telah disimpan sebagai 'cat_breed_classifier_final_mobilenet.keras'")