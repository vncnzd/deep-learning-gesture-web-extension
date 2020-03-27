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
        this.numberOfExamples = 0;

        this.webcam = new Webcam(this.videoElement);
        this.data = new DataContainer();
        this.model = new Model(2);

        this.addEventListeners();
    }

    addEventListeners() {
        let self = this;

        this.captureImageButtonElement.addEventListener('click', function() {
            let imageTensor = self.webcam.captureImageAndGetTensor();
            tf.browser.toPixels(imageTensor, self.imagePreviewCanvasElement)

            let expandedImageTensor = tf.expandDims(imageTensor, 0);
            let currentFeature = self.featureSelectElement.options[self.featureSelectElement.selectedIndex].value;
            let currentFeatureTensor = tf.tidy(() => tf.oneHot(tf.tensor1d([parseInt(currentFeature)]).toInt(), 2));
            
            self.data.add(expandedImageTensor, currentFeatureTensor);
            self.examplesSpanElement.innerHTML = ++self.numberOfExamples;
        });

        this.trainNetworkButtonElement.addEventListener('click', function() {
            self.model.train(self.data.xTrain, self.data.yTrain, 10);
        });
    }
}

let app = new App();

