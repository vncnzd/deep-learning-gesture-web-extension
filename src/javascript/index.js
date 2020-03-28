import * as tf from '@tensorflow/tfjs';
import '../sass/style.scss';
import Webcam from './webcam.js';
import DataContainer from './dataContainer';
import Model from './model';

class App {
    constructor() {
        this.captureImageButtonElement = document.querySelector("#capture-image-button");
        this.trainNetworkButtonElement = document.querySelector("#train-network-button");
        this.videoElement = document.querySelector('#webcam-video');
        this.imagePreviewCanvasElement = document.querySelector('#image-preview-canvas');
        this.featureSelectElement = document.querySelector('#feature-select');
        this.examplesSpanElement = document.querySelector("#examples-span");
        this.predictButtonElement = document.querySelector("#predict-button");
        this.numberOfEpochsInputElement = document.querySelector("#number-of-epochs-input");
        this.loadingBarElement = document.querySelector("#training-loading-bar");
        this.lossSpanElement = document.querySelector("#loss-span");

        this.numberOfExamples = 0;
        this.loadingBarProgress = 0;
        this.webcam = new Webcam(this.videoElement, 50);
        this.data = new DataContainer();
        this.model = new Model(50, 50, 3, 4);

        this.addEventListeners();
    }

    addEventListeners() {
        this.captureImageButtonElement.addEventListener('click', () => {
            let imageTensor = this.webcam.captureImageAndGetTensor();
            tf.browser.toPixels(imageTensor, this.imagePreviewCanvasElement)

            let expandedImageTensor = tf.expandDims(imageTensor, 0);
            let currentFeature = this.featureSelectElement.options[this.featureSelectElement.selectedIndex].value;
            let currentFeatureTensor = tf.tidy(() => tf.oneHot(tf.tensor1d([parseInt(currentFeature)]).toInt(), 4));

            this.data.add(expandedImageTensor, currentFeatureTensor);
            this.examplesSpanElement.innerHTML = ++this.numberOfExamples;
        });

        this.trainNetworkButtonElement.addEventListener('click', () => {
            let numberOfEpochs = parseInt(this.numberOfEpochsInputElement.value);
            this.model.train(this.data.xTrain, this.data.yTrain, numberOfEpochs, this.updateTrainingProgress.bind(this));
        });

        this.predictButtonElement.addEventListener('click', () => {
            let imageTensor = this.webcam.captureImageAndGetTensor();
            let resultTensor = tf.tidy(() => { return this.model.predict(imageTensor.expandDims(0)) });

            resultTensor.print();

            tf.dispose(imageTensor);
            tf.dispose(resultTensor);
        });
    }

    updateTrainingProgress(batch, logs) {
        this.lossSpanElement.innerHTML = logs.loss.toFixed(7);

        let numberOfEpochs = parseInt(this.numberOfEpochsInputElement.value);
        let progressStep = 100 / numberOfEpochs;

        this.loadingBarProgress += progressStep;
        this.loadingBarElement.style.width = this.loadingBarProgress + "%";
    }
}

let app = new App();

