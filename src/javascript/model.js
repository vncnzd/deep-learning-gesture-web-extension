import * as tf from '@tensorflow/tfjs';

class Model {
    constructor(numberOfOutputClasses) {
        this.imageHeight = 224;
        this.imageWidth = 224;
        this.imageChannels = 3;
        this.numberOfOutputClasses = numberOfOutputClasses;
        this.model = tf.sequential();

        this.addFirstConvolutionalLayer(this.model);
        this.addFirstMaxPoolingLayer(this.model);
        this.addSecondConvolutionalLayer(this.model);
        this.addSecondMaxPoolingLayer(this.model);
        this.addFlattenLayer(this.model);
        this.addDenseLayer(this.model);
        this.compileModel(this.model);
    }

    addFirstConvolutionalLayer(model) {
        model.add(tf.layers.conv2d({
            inputShape: [this.imageWidth, this.imageWidth, this.imageChannels],
            kernelSize: 5,
            filters: 8,
            strides: 1,
            activation: 'relu',
            kernelInitializer: 'varianceScaling'
        }));
    }

    addFirstMaxPoolingLayer(model) {
        model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
    }

    addSecondConvolutionalLayer(model) {
        model.add(tf.layers.conv2d({
            kernelSize: 5,
            filters: 16,
            strides: 1,
            activation: 'relu',
            kernelInitializer: 'varianceScaling'
        }));
    }

    addSecondMaxPoolingLayer(model) {
        model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
    }

    addFlattenLayer(model) {
        model.add(tf.layers.flatten());
    }

    addDenseLayer(model) {
        model.add(tf.layers.dense({
            units: this.numberOfOutputClasses,
            kernelInitializer: 'varianceScaling',
            activation: 'softmax'
        }));
    }

    compileModel(model) {
        model.compile({
            optimizer: tf.train.adam();,
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy'],
        });
    }

    getModel() {
        return this.model;
    }
}