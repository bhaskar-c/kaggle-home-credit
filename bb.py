import pandas as pd
import gc
from utils import *


class BureauBalance:

  def __init__(self, debug=False):
    self.bb = reduce_mem_usage(read_data('bureau_balance', debug=debug))
    self.df = pd.Datarame()


  def execute(self):
    self.clean()
    df1 = self.handcrafted()
    df2 = self.aggregations()
    df = df1.append(df2)
    return df

  def clean():
    pass

  def handcrafted(self):
    df = self.tr_te[0:5]
    return df

  def aggregations(self):
    df = self.tr_te[6:9]
    return df

