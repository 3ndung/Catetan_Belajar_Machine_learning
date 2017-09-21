import pandas as pd
from keras.models import Sequential
from keras.layers import  *

Data_Belajar = pd.read_csv('Finale_Training.csv')
print Data_Belajar.head(1)

X = Data_Belajar.drop('LEVEL_KEMACETAN', axis=1).values
Y = Data_Belajar[['LEVEL_KEMACETAN']].values

Data_uji = pd.read_csv('Finale_Test.csv')

X_uji = Data_uji.drop(['LEVEL_KEMACETAN'], axis=1).values
Y_uji = Data_uji[['LEVEL_KEMACETAN']].values

X_predict = pd.read_csv('predict.csv', sep='~')
X_predict = X_predict.drop(['WILAYAH','NAMA_JALAN','TAHUN','BULAN','HARI','LEVEL_KEMACETAN'], axis=1)
X_predict = X_predict.head(1)

print(" ini data beda ")
print X
print(" ini data beda juga")
print Y
print(" ini juga beda ")
print X_predict


model = Sequential()
model.add(Dense(50, input_dim=4, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error',optimizer='adam')

model.fit(
    X,
    Y,
    epochs=1000,
    shuffle=True,
    verbose=2
)

Test_Erroe_Rate = model.evaluate(
    X_uji,
    Y_uji,
    verbose=0
)

print("ini Adalah nilai Test Error Rate nya {}".format(Test_Erroe_Rate))
print(Test_Erroe_Rate)

model.save('New_Model_Learn.h5')
print("Model Saved ")