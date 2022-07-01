import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os, io

def PLOT_PREDICTION(x, xl='mTT [GeV]'):
  fig = plt.figure()
  plt.xlabel(xl)
  plt.xlim(50, 250)
  plt.hist(x, bins=100, range=(50, 250))
  return fig

def PLOT_PREDICTION_VS_TARGET(x, y, xl='SVFit [GeV]', yl='mTT [GeV]'):
  reg = linregress(list(x), list(y))
  aa, bb = reg.slope, reg.intercept
  fig = plt.figure()
  plt.xlim(50, 250)
  plt.ylim(50, 250)
  plt.xlabel(xl)
  plt.ylabel(yl)
  plt.plot([50,250], [aa*50+bb,aa*250+bb], color='blue', marker='', label='fit')
  plt.plot([50,250], [50,250]            , color='red' , marker='', label='target value')
  plt.scatter(x, y, marker='.', s=1)
  plt.legend(loc='upper left')
  return fig

class mTTPlotCallback(keras.callbacks.Callback):
  def __init__(self, name, data, target, log_dir):
    super().__init__()
    self.n = name
    self.x = data
    self.y = target
    self.log_dir = log_dir

    if not os.path.exists(self.log_dir):
      os.makedirs(self.log_dir)
    mTTPlotCallback.writer=tf.summary.create_file_writer(self.log_dir)
  #def on_batch_begin          (self, step, log={}): return
  #def on_batch_end            (self, step, log={}): return
  #def on_epoch_begin          (self, step, log={}): return
  #def on_predict_batch_begin  (self, step, log={}): return
  #def on_predict_batch_end    (self, step, log={}): return
  #def on_predict_begin        (self, step, log={}): return
  #def on_predict_end          (self, step, log={}): return
  #def on_test_batch_begin     (self, step, log={}): return
  #def on_test_batch_end       (self, step, log={}): return
  #def on_test_begin           (self, step, log={}): return
  #def on_test_end             (self, step, log={}): return
  #def on_train_batch_begin    (self, step, log={}): return
  #def on_train_batch_end      (self, step, log={}): return
  #def on_train_begin          (self, step, log={}): return
  def on_train_end(self, step, log={}):
    mTTPlotCallback.writer.close()
    prediction=self.model.predict(self.x).reshape(len(self.x))
    plot_img = PLOT_PREDICTION           (x=prediction)
    scat_img = PLOT_PREDICTION_VS_TARGET (y=prediction, x=self.y.values)
    plot_img.savefig(self.log_dir+"/massplot_{}.pdf".format(self.n))
    scat_img.savefig(self.log_dir+"/scatterplot_{}.pdf".format(self.n))

  def on_epoch_end(self, epoch, logs={}):
    prediction=self.model.predict(self.x).reshape(len(self.x))
    plot_img = self.__img_to_tf(PLOT_PREDICTION           (x=prediction))
    scat_img = self.__img_to_tf(PLOT_PREDICTION_VS_TARGET (y=prediction, x=self.y.values))

    with mTTPlotCallback.writer.as_default(step=epoch):
      tf.summary.image("HH mass - {}"      .format(self.n), plot_img, step=epoch)
      tf.summary.image("mTT vs. SVFit - {}".format(self.n), scat_img, step=epoch)

  @staticmethod
  def __img_to_tf(fig):
    with io.BytesIO() as buffer:
      fig.savefig(buffer, format='png')
      img = tf.image.decode_png(buffer.getvalue(), channels=4)
      img = tf.expand_dims(img, 0)
    return img
      