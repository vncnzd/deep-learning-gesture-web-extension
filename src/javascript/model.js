import * as tf from '@tensorflow/tfjs';

class Model {
    constructor(imageWidth, imageHeight, imageChannels, numberOfOutputClasses, storageDirectory) {
        this.imageWidth = imageWidth;
        this.imageHeight = imageHeight;
        this.imageChannels = imageChannels;
        this.numberOfOutputClasses = numberOfOutputClasses;
        this.model = tf.sequential();
        this.storageDirectory = storageDirectory;

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
            optimizer: tf.train.adam(),
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy'],
        });
    }

    async train(imageTensors, labelTensors, numberOfEpochs, epochCallback) {
        await this.model.fit(imageTensors, labelTensors, {
            batch_size: 20,
            epochs: numberOfEpochs,
            callbacks: {
                onEpochEnd: epochCallback,
            }
        });

        this.model.save(this.storageDirectory);
    }

    async load(storageDirectory) {
        let model = await tf.loadLayersModel(storageDirectory).then(() => { 
            console.log("Loading model successful"); 
        }).catch(() => {
            console.log("No model to load");
        });
        
        if (model != null) {
            this.model = model;
        }
    }

    predict(imageTensor) {
        return this.model.predict(imageTensor);
    }
}

export default Model;