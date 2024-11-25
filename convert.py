import tensorflow as tf

model = tf.keras.models.load_model("model_deteksi_penyakit_ayam.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("model_deteksi_penyakit_ayam.tflite", "wb") as f:
    f.write(tflite_model)
