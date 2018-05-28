import tensorflow as tf

from imageGen import GenerateImages


def main():
    hello = tf.constant('Hello, TensorFlow!')
    sess = tf.Session()
    print(sess.run(hello))
    images = GenerateImages();
    print(images);












main();