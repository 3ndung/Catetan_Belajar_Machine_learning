import pandas as pd
from sklearn.preprocessing import MinMaxScaler

Data_training = pd.read_csv('traffic_training_set.csv', sep='\t')
Data_training = Data_training.dropna()
Data_training = Data_training.drop(['WILAYAH','NAMA_JALAN','TAHUN','BULAN','HARI'],axis=1)
Data_training = Data_training.head(2000)
Data_test = Data_training.head(700)

#Data_test = pd.read_csv('traffic_test_set.csv', sep='\t')
#Data_test = Data_test.dropna()
#Data_test = Data_test.drop(['WILAYAH','NAMA_JALAN','TAHUN','BULAN'],axis=1)
#Data_test = Data_test.head(700)
#print "Biar gak Lupa urutanya Aja"
print Data_training.head(1)
print Data_test.head(1)

scaler = MinMaxScaler(feature_range=(0,1))

Data_training_scale = scaler.fit_transform(Data_training)
Data_test_scale = scaler.transform(Data_test)

print "Berikut adalah Nilai dari Level_Kemacetan dikali {:.10f} & dibagi {:.6f}".format(scaler.scale_[4], scaler.min_[4])

Data_training_scale_df = pd.DataFrame(Data_training_scale, columns=Data_training.columns.values)
Data_test_scale_df = pd.DataFrame(Data_test_scale, columns=Data_test.columns.values)

print "Next kita akan mengconvert data yang sudah discale ke dalam .csv"

Data_training_scale_df.to_csv('Finale_Training.csv',index=False)
Data_test_scale_df.to_csv('Finale_Test.csv', index=False)
Data_test_scale_df.head(1).drop('LEVEL_KEMACETAN',axis=1).to_csv('New_Dugaan.csv',index=False)

