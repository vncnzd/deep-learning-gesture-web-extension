import * as tf from '@tensorflow/tfjs';
import '../sass/style.scss';
import Webcam from './webcam.js';
import DataContainer from './dataContainer';
import Model from './model';
import ElementsManager from './elementsManager';

class App {
    constructor() {
        this.activationThreshold = 0.9;
        this.numberOfExamples = 0;
        this.loadingBarProgress = 0;

        this.elementsManager = new ElementsManager();
        this.webcam = new Webcam(this.elementsManager.videoElement, 50);

        this.dataContainer = new DataContainer();
        this.dataContainer.localStorageKey = "training-data";
        
        this.modelStorageName = "gesture-extension-model";
        this.modelStorageDirectory = 'localstorage://' + this.modelStorageName;
        this.model = new Model(50, 50, 3, 4, this.modelStorageDirectory);
        this.model.load(this.model.storageDirectory);

        this.addEventListeners();
    }

    addEventListeners() {
        this.elementsManager.captureImageButtonElement.addEventListener('click', () => {
            let imageTensor = this.webcam.captureImageAndGetTensor();
            tf.browser.toPixels(imageTensor, this.elementsManager.imagePreviewCanvasElement);

            let expandedImageTensor = tf.expandDims(imageTensor, 0);
            let currentFeature = this.elementsManager.featureSelectElement.options[this.elementsManager.featureSelectElement.selectedIndex].value;
            let currentFeatureTensor = tf.tidy(() => tf.oneHot(tf.tensor1d([parseInt(currentFeature)]).toInt(), 4));

            this.dataContainer.add(expandedImageTensor, currentFeatureTensor);
            this.elementsManager.numberOfExamplesElement.innerHTML = ++this.numberOfExamples;
        });

        this.elementsManager.trainNetworkButtonElement.addEventListener('click', async () => {
            let numberOfEpochs = parseInt(this.elementsManager.numberOfEpochsInputElement.value);
            await this.model.train(this.dataContainer.xTrain, this.dataContainer.yTrain, numberOfEpochs, this.updateTrainingProgress.bind(this));
            this.makePredictionEverySeconds(2)
        });

        this.elementsManager.removeModelButtonElement.addEventListener('click', () => {
            Object.keys(localStorage).forEach(key => {
                if (key.includes(this.modelStorageName)) {
                    localStorage.removeItem(key);
                }
            });

            console.log(Object.keys(localStorage));
        });

        this.elementsManager.saveImagesButtonElement.addEventListener('click', () => {
            this.dataContainer.save();
        });

        this.elementsManager.loadImagesButtonElement.addEventListener('click', () => {
            this.dataContainer.load().then(() => {
                if (this.dataContainer.xTrain != null) {
                    this.numberOfExamples = this.dataContainer.xTrain.shape[0];
                    this.elementsManager.numberOfExamplesElement.innerHTML = this.numberOfExamples;
                }
            });
        });

        this.elementsManager.removeImageButtonElement.addEventListener('click', () => {
            this.dataContainer.remove().then(() => {
                this.numberOfExamples = 0;
                this.elementsManager.numberOfExamplesElement.innerHTML = this.numberOfExamples;
            });
        });
    }

    updateTrainingProgress(batch, logs) {
        this.lossSpanElement.innerHTML = logs.loss.toFixed(7);

        let numberOfEpochs = parseInt(this.elementsManager.numberOfEpochsInputElement.value);
        let progressStep = 100 / numberOfEpochs;

        this.loadingBarProgress += progressStep;
        this.elementsManager.loadingBarElement.style.width = this.loadingBarProgress + "%";
    }

    makePredictionEverySeconds(numberOfSeconds) {
        let milliseconds = numberOfSeconds * 1000;
        window.setInterval(this.makePrediction.bind(this), milliseconds);
    }

    makePrediction() {
        let imageTensor = this.webcam.captureImageAndGetTensor();
        let resultTensor = tf.tidy(() => { return this.model.predict(imageTensor.expandDims(0)) });

        this.evaluatePredictionTensor(resultTensor);

        tf.dispose(imageTensor);
        tf.dispose(resultTensor);
    }

    evaluatePredictionTensor(predictionTensor) {
        let maxPredictionValue = predictionTensor.max().dataSync()[0];
        console.log("Max prediction value: " + maxPredictionValue);

        if (maxPredictionValue < this.activationThreshold) return;

        let label = predictionTensor.argMax(1).dataSync()[0];
        console.log("label: " + label);

        this.executeAction(label);
    }

    executeAction(label) {
        switch (label) {
            case 1:
                browser.tabs.create({
                    active: true
                });
                break;
            case 2:
                browser.tabs.query({currentWindow: true, active: true}).then(function(tabs) {
                    if (tabs.length > 0) {
                        browser.tabs.remove(tabs[0].id);
                    }
                });
                break;
            case 3:
                browser.tabs.query({currentWindow: true, active: true}).then(function(tabs) {
                    if (tabs.length > 0) {
                        browser.tabs.reload(tabs[0].id);
                    }
                });
                break;
            default:
                break;
        }
    }
}

let app = new App();
