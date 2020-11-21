def base_model_plot(history, model_name, save_path=None):
  title = model_name
  pylab.figure(figsize=(12,4))
  pylab.subplot(1,2,1)
  if 'acc' in history.keys():
    pylab.plot(history['acc'], label='train')
    pylab.plot(history['val_acc'], label='validation')
  else:
    pylab.plot(history['accuracy'], label='train')
    pylab.plot(history['val_accuracy'], label='validation')
  pylab.title('Model Accuracy for {}'.format(title))
  pylab.xlabel('epochs')
  pylab.ylabel('accuracy')
  pylab.legend(loc='best')

  pylab.subplot(1,2,2)
  pylab.plot(history['loss'], label='train')
  pylab.plot(history['val_loss'], label='validation')
  pylab.title('Model Loss for {}'.format(title))
  pylab.xlabel('epochs')
  pylab.ylabel('loss')
  pylab.legend(loc='best')
  
  if save_path:
    pylab.savefig('{}/{}.pdf'.format(save_path, title))