#Red neuronal basica 
#1 neurona, 1 capa
#pip install -r requirements.txt
#En cmd jupyter notebook --notebook-dir=D:
model =  tf.keras.Sequential([
    tf.keras.Input(shape=(1,)), #entrada
    tf.keras.layers.Dense(units=1) #1era capa, 1 neurona
])

#definicion de funciones
#las dos basicas, loss function y optimizer
model.compile(optimizer='sgd',loss='mean_squared_error')

#Datos
xs = np.array([-1.0,0.0,1.0,2.0,3.0,4.0],dtype=float)
ys = np.array([-3.0,-1.0,1.0,3.0,5.0,7.0],dtype=float)

#entrenar
model.fit(xs,ys,epochs=500) 

#inferencia
print(model.predict([10.0]))