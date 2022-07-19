import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os, io
import numpy as np

def PLOT_PREDICTION(x1, x2=None, l1='mTT', l2='SVFit', xl='$\\tau\\tau$ mass [GeV]', lo=50, hi=250):
  fig = plt.figure()
  plt.xlabel(xl)
  plt.xlim(lo, hi)
  plt.hist(x1, bins=100, range=(lo, hi), label=l1)
  if x2 is not None:
    plt.hist(x2, bins=100, range=(lo, hi), color='red', alpha=0.3, label=l2)
  plt.legend(loc='upper right')
  return fig

def PLOT_PREDICTION_VS_TARGET(x, y, xl='SVFit [GeV]', yl='mTT [GeV]'):
  pol1 = lambda a,b,x  : a*x+b
  reg = linregress(list(x), list(y))
  p1a, p1b = reg.slope, reg.intercept
  fig = plt.figure()
  plt.xlim(50, 250)
  plt.ylim(50, 250)
  plt.xlabel(xl)
  plt.ylabel(yl)
  plt.plot([50,250], [pol1(p1a,p1b    ,50), pol1(p1a,p1b    ,250)], color='blue' , marker='', label='lin. fit')
  plt.plot([50,250], [pol2(p2a,p2b,p2c,50), pol2(p2a,p2b,p2c,250)], color='green', marker='', label='pol2 fit')
  plt.plot([50,250], [50,250]                                     , color='red'  , marker='', label='target value')
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

  def on_train_end(self, step, log={}):
    mTTPlotCallback.writer.close()
    prediction=self.model.predict(self.x).reshape(len(self.x))
    plot_img = PLOT_PREDICTION           (x1=prediction, x2=self.y.values)
    diff_img = PLOT_PREDICTION           (x1=(prediction-self.y.values)/self.y.values, xl='mTT/SVFit-1', lo=-.5, hi=.5)
    scat_img = PLOT_PREDICTION_VS_TARGET (y=prediction, x=self.y.values)
    plot_img.savefig(self.log_dir+"/massplot_{}.pdf"    .format(self.n.replace('/', '_')))
    scat_img.savefig(self.log_dir+"/scatterplot_{}.pdf" .format(self.n.replace('/', '_')))
    diff_img.savefig(self.log_dir+"/diff_{}.pdf"        .format(self.n.replace('/', '_')))

  def on_epoch_end(self, epoch, logs={}):
    prediction=self.model.predict(self.x).reshape(len(self.x))
    plot_img = self.__img_to_tf(PLOT_PREDICTION           (x1=prediction, x2=self.y.values))
    diff_img = self.__img_to_tf(PLOT_PREDICTION           (x1=(prediction-self.y.values)/self.y.values, xl='mTT/SVFit-1', lo=-.5, hi=.5))
    scat_img = self.__img_to_tf(PLOT_PREDICTION_VS_TARGET (y=prediction, x=self.y.values))

    with mTTPlotCallback.writer.as_default(step=epoch):
      tf.summary.image("HH_mass/{}" .format(self.n), plot_img, step=epoch)
      tf.summary.image("diff/{}"    .format(self.n), diff_img, step=epoch)
      tf.summary.image("scatter/{}" .format(self.n), scat_img, step=epoch)

  @staticmethod
  def __img_to_tf(fig):
    with io.BytesIO() as buffer:
      fig.savefig(buffer, format='png')
      img = tf.image.decode_png(buffer.getvalue(), channels=4)
      img = tf.expand_dims(img, 0)
    return img
      