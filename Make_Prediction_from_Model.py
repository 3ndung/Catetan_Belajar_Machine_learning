import pandas as pd
from keras.models import load_model


mod = load_model("New_Model_Learn.h5")

X = pd.read_csv('New_Dugaan.csv').values
#X = X.drop(['WILAYAH','NAMA_JALAN','TAHUN','BULAN','HARI','LEVEL_KEMACETAN'], axis=1)
print X.shape
print X
#Dugaan = model.predict(X)
Dugaan = mod.predict(X)


Dugaan = Dugaan[0][0]

#dugaan = dugaan + nilai min dari object prediki feature training
#dugaan = dugaan / nilai skala prediksi feature training

Dugaan = Dugaan + 1.008380
Dugaan = Dugaan / 0.6383028136


print("LEVEL_KEMACETAN yang di prediksi dari kondisi yang dimasukan adalah : {} ".format(Dugaan))