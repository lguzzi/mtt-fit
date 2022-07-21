import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os, io
import numpy as np
import ROOT

def PLOT_PREDICTION(x, xl, lo, hi):
  fig = plt.figure()
  plt.xlabel(xl)
  plt.xlim(lo, hi)
  plt.hist(x, bins=100, range=(lo, hi))
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

  def on_train_end(self, step, log={}):
    mTTPlotCallback.writer.close()
    prediction = self.model.predict(self.x)
    difference = (prediction - self.y.values) / self.y.values

    plot_px = PLOT_PREDICTION(x=[x[0] for x in difference], xl='(pX - genX)/genX', lo=-2, hi=2)
    plot_py = PLOT_PREDICTION(x=[x[1] for x in difference], xl='(pY - genY)/genY', lo=-2, hi=2)
    plot_pz = PLOT_PREDICTION(x=[x[2] for x in difference], xl='(pZ - genZ)/genZ', lo=-2, hi=2)
    plot_m  = PLOT_PREDICTION(x=[x[3] for x in difference], xl='(M - genM)/genM' , lo=-2, hi=2)
    
    plot_img.savefig(self.log_dir+"/diff_Px{}.pdf".format(self.n.replace('/', '_')))
    plot_img.savefig(self.log_dir+"/diff_Py{}.pdf".format(self.n.replace('/', '_')))
    plot_img.savefig(self.log_dir+"/diff_Pz{}.pdf".format(self.n.replace('/', '_')))
    plot_img.savefig(self.log_dir+"/diff_M{}.pdf" .format(self.n.replace('/', '_')))

  def on_epoch_end(self, epoch, logs={}):
    prediction = self.model.predict(self.x)
    difference = (prediction - self.y.values) / self.y.values
    

    plot_px = self.__img_to_tf(PLOT_PREDICTION(x=[x[0] for x in difference], xl='(pX - genX)/genX', lo=-2, hi=2))
    plot_py = self.__img_to_tf(PLOT_PREDICTION(x=[x[1] for x in difference], xl='(pY - genY)/genY', lo=-2, hi=2))
    plot_pz = self.__img_to_tf(PLOT_PREDICTION(x=[x[2] for x in difference], xl='(pZ - genZ)/genZ', lo=-2, hi=2))
    plot_m  = self.__img_to_tf(PLOT_PREDICTION(x=[x[3] for x in difference], xl='(M - genM)/genM' , lo=-2, hi=2))

    with mTTPlotCallback.writer.as_default(step=epoch):
      tf.summary.image("diff_Px/{}".format(self.n), plot_px, step=epoch)
      tf.summary.image("diff_Py/{}".format(self.n), plot_py, step=epoch)
      tf.summary.image("diff_Pz/{}".format(self.n), plot_pz, step=epoch)
      tf.summary.image("diff_M/{}" .format(self.n), plot_m , step=epoch)

  @staticmethod
  def __img_to_tf(fig):
    with io.BytesIO() as buffer:
      fig.savefig(buffer, format='png')
      img = tf.image.decode_png(buffer.getvalue(), channels=4)
      img = tf.expand_dims(img, 0)
    return img
      