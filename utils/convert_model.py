
for i, layer in enumerate(model.layers):
    print('i ',i, layer.name)
    if i > 8 and i != 45 and i <= 234: 
        model1.get_layer(name = layer.name).set_weights(layer.get_weights())
model1.save('test.h5')

