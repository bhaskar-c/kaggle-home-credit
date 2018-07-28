from tr_te import TrainTest
from b import Bureau
from cc import CreditCard
from ip import InstallmentPayments
from pa import PreviousApplication
from pos import POSCASHBalance
import gc

debug = True

def get_consolidated_table():
  df = TrainTest(debug=debug).execute()
  b = Bureau(debug=debug).execute()
  df = df.merge(b, on=['SK_ID_CURR'], how='left')
  cc = CreditCard(debug=debug).execute()
  df = df.merge(cc, on=['SK_ID_CURR'], how='left')
  ip = InstallmentPayments(debug=debug).execute()
  df = df.merge(ip, on=['SK_ID_CURR'], how='left')
  pa = PreviousApplication(debug=debug).execute()
  df = df.merge(pa, on=['SK_ID_CURR'], how='left')
  pos = POSCASHBalance(debug=debug).execute()
  df = df.merge(pos, on=['SK_ID_CURR'], how='left')
  del (b,cc,ip,pa, pos)
  gc.collect()
  return df

df = get_consolidated_table()
print(df.shape)


# woe
# target encoding
# categorical embedding vith neural network
# libfm, vowpal vabbit
# multilayer stacking
