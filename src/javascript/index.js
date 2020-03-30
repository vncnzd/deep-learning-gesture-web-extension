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
        this.currentImagePreviewIndex = 0;

        this.elementsManager = new ElementsManager();
        this.webcam = new Webcam(this.elementsManager.videoElement, 50);

        this.dataContainer = new DataContainer();
        this.dataContainer.localStorageKey = "training-data";
        this.dataContainer.load().then(this.changeImageStack.bind(this));
        
        this.modelStorageName = "gesture-extension-model";
        this.model = new Model(50, 50, 3, 4, this.modelStorageName);
        this.model.load(this.model.storageDirectory);

        this.addEventListeners();
    }

    addEventListeners() {
        this.elementsManager.captureImageButtonElement.addEventListener('click', this.captureImage.bind(this));
        this.elementsManager.trainNetworkButtonElement.addEventListener('click', this.trainModel.bind(this));
        this.elementsManager.removeModelButtonElement.addEventListener('click', this.removeModelFromStorage.bind(this));
        this.elementsManager.saveImagesButtonElement.addEventListener('click', () => { this.dataContainer.save(); });
        this.elementsManager.loadImagesButtonElement.addEventListener('click', this.loadDataContainer.bind(this));
        this.elementsManager.removeImagesButtonElement.addEventListener('click', this.removeAllImageTensors.bind(this));

        this.elementsManager.nextImageButtonElement.addEventListener('click', () => { this.cycleThroughImages(1); });
        this.elementsManager.previousImageButtonElement.addEventListener('click', () => { this.cycleThroughImages(-1); });

        this.elementsManager.removeImageButtonElement.addEventListener('click', this.removeCurrentImage.bind(this));
        this.elementsManager.featureSelectElement.addEventListener('input', this.changeImageStack.bind(this));
    }

    loadDataContainer() {
        this.dataContainer.load().then(() => {
            if (this.dataContainer.xTrain != null) {
                let currentFeatureLabel = this.getCurrentFeatureLabel();
                this.loadImageWithIndexAndLabelIntoCanvas(this.currentImagePreviewIndex, currentFeatureLabel, this.elementsManager.imagePreviewCanvasElement);
                this.setExampleIndicator(currentFeatureLabel, this.elementsManager.numberOfExamplesElement);
            }
        });
    }

    removeCurrentImage() {
        let currentFeatureLabel = this.getCurrentFeatureLabel();
        this.dataContainer.removeTensorFromBatch(currentFeatureLabel, this.currentImagePreviewIndex);
        this.cycleThroughImages(-1);
        this.setExampleIndicator(currentFeatureLabel, this.elementsManager.numberOfExamplesElement);
    }

    changeImageStack() {
        let currentFeatureLabel = this.getCurrentFeatureLabel();
        this.currentImagePreviewIndex = 0;
        this.loadImageWithIndexAndLabelIntoCanvas(0, currentFeatureLabel, this.elementsManager.imagePreviewCanvasElement);
        this.setExampleIndicator(currentFeatureLabel, this.elementsManager.numberOfExamplesElement);
    }

    cycleThroughImages(step) {
        let currentFeatureLabel = this.getCurrentFeatureLabel();
        this.currentImagePreviewIndex += step;

        if (this.currentImagePreviewIndex < 0) {
            this.currentImagePreviewIndex = 0;
        } else if (this.currentImagePreviewIndex >= this.dataContainer.getNumberOfTensorsForLabel(currentFeatureLabel)) {
            // Refactor this, since getNumberOfTensorsForLabel is an expensive method.
            this.currentImagePreviewIndex = this.dataContainer.getNumberOfTensorsForLabel(currentFeatureLabel) - 1;
        }

        this.loadImageWithIndexAndLabelIntoCanvas(this.currentImagePreviewIndex, currentFeatureLabel, this.elementsManager.imagePreviewCanvasElement);
        this.setExampleIndicator(currentFeatureLabel, this.elementsManager.numberOfExamplesElement);
    }

    removeAllImageTensors() {
        this.dataContainer.removeAllTensorsFromStorage().then(() => {
            console.log("Removing all tensors successful");
        });
    }

    captureImage() {
        let imageTensor = this.webcam.captureImageAndConvertToTensor();
        tf.browser.toPixels(imageTensor, this.elementsManager.imagePreviewCanvasElement);

        let expandedImageTensor = tf.expandDims(imageTensor, 0);
        let currentFeatureLabel = this.getCurrentFeatureLabel();
        let currentFeatureTensor = tf.tidy(() => tf.oneHot(tf.tensor1d([parseInt(currentFeatureLabel)]).toInt(), 4));
        this.dataContainer.add(expandedImageTensor, currentFeatureTensor);

        this.currentImagePreviewIndex = this.dataContainer.getNumberOfTensorsForLabel(currentFeatureLabel) - 1;
        this.setExampleIndicator(currentFeatureLabel, this.elementsManager.numberOfExamplesElement);
    }

    removeModelFromStorage() {
        this.model.removeFromStorage();
    }

    async trainModel() {
        let numberOfEpochs = parseInt(this.elementsManager.numberOfEpochsInputElement.value);
        await this.model.train(this.dataContainer.xTrain, this.dataContainer.yTrain, numberOfEpochs, this.updateTrainingProgress.bind(this));
        this.makePredictionEverySeconds(2);
    }

    getCurrentFeatureLabel() {
        return this.elementsManager.featureSelectElement.options[this.elementsManager.featureSelectElement.selectedIndex].value;
    }

    setExampleIndicator(label, element) {
        let numberOfTensors = this.dataContainer.getNumberOfTensorsForLabel(label);
        let indicator = (this.currentImagePreviewIndex + 1) + "/" + numberOfTensors;
        element.innerHTML = indicator;
    }

    loadImageWithIndexAndLabelIntoCanvas(index, label, canvasElement) {
        let imageData = this.dataContainer.getTensorDataForYLabel(label, index);
        tf.browser.toPixels(tf.tensor3d(imageData), canvasElement);
    }

    updateTrainingProgress(batch, logs) {
        this.elementsManager.lossSpanElement.innerHTML = logs.loss.toFixed(7);

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
        let imageTensor = this.webcam.captureImageAndConvertToTensor();
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
