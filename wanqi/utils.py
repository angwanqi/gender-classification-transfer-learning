# import libraries
from keras.callbacks import ModelCheckpoint

def generate_data(train_df, val_df, test_df, img_size, batch_size):
  # data generator for training data
  train_generator = train_datagen.flow_from_dataframe(train_df, x_col="path", 
                                                      y_col="gender", 
                                                      target_size=(img_size,img_size), 
                                                      batch_size=batch_size, 
                                                      class_mode='binary')

  # data generator for validation data
  validation_generator = validation_datagen.flow_from_dataframe(val_df, x_col="path", 
                                                                y_col="gender", 
                                                                target_size=(img_size,img_size),
                                                                batch_size=batch_size, 
                                                                class_mode='binary')

  # data generator for validation data to be used for prediction
  predict_generator = validation_datagen.flow_from_dataframe(val_df, x_col="path", 
                                                             y_col="gender", 
                                                             target_size=(img_size,img_size),
                                                             batch_size=batch_size, 
                                                             class_mode='binary',
                                                             shuffle = False)

  # data generator for testing data
  test_generator = test_datagen.flow_from_dataframe(test_df, x_col="path", 
                                                    y_col="gender", 
                                                    target_size=(img_size,img_size),
                                                    batch_size=batch_size, 
                                                    class_mode='binary',
                                                    shuffle = False)
  
  generators = {'train_gen': train_generator,
                'validation_gen': validation_generator,
                'test_gen': test_generator,
                'predict_gen': predict_generator}
  
  return generators


 # Define Model Checkpoint callback
def mc(title, save_path):
  file_path = '{}/{}_mc.h5'.format(save_path, title)
  return ModelCheckpoint(file_path, monitor='val_acc', mode='max', verbose=1, save_best_only=True)

def fit(model, epochs, title, generator, es, save_path):
  train_gen = generator['train_gen']
  validation_gen = generator['validation_gen']
  history = model.fit(train_gen, 
                      steps_per_epoch=train_gen.samples/train_gen.batch_size, 
                      epochs=epochs,
                      validation_data=validation_gen,
                      validation_steps=validation_gen.samples/validation_gen.batch_size,
                      verbose=2,
                      callbacks=[es, mc(title, save_path)])

  save_history(history, save_path, title)

  return history

  def save_history(history, save_path, title):
  # Saving history to file
  filename = '{}/{}_hist.pickle'.format(save_path, title)
  with open(filename, 'wb') as file_pi:
      pickle.dump(history.history, file_pi)
  print("Successfully saved history file at {}".format(filename))