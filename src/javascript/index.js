import * as tf from '@tensorflow/tfjs';
import '../sass/style.scss';
import Webcam from './webcam.js';
import DataContainer from './dataContainer';
import Model from './model';
import ElementsManager from './elementsManager';
import LocalStorage from './localStorage';

class App {
    constructor() {
        this.elementsManager = new ElementsManager();
        this.activationThreshold = 0.9;
        this.loadingBarProgress = 0;
        this.currentImagePreviewIndex = 0;
        this.shouldMakePredictions = false;
        this.imageSize = 60;
        
        this.loadShouldMakePredictions().then(this.setPredictionButtonsColors.bind(this));
        this.loadActivationThreshold();

        this.webcam = new Webcam(this.elementsManager.videoElement, this.imageSize);
        this.features = this.elementsManager.getOptionsOfSelectElement(this.elementsManager.featureSelectElement);

        this.dataContainer = new DataContainer();
        this.dataContainer.localStorageKey = "training-data";
        this.dataContainer.load().then(this.changeImageStack.bind(this)).catch((error) => { 
            this.elementsManager.setMessageAndLog(error);
        });
        
        this.modelStorageName = "gesture-extension-model";
        this.model = new Model(this.imageSize, this.imageSize, 3, Object.keys(this.features).length, this.modelStorageName);
        this.model.load(this.model.storageDirectory);

        this.addEventListeners();

        this.makePredictionEverySeconds(1);
        this.setPredictionButtonsColors();
    }

    addEventListeners() {
        this.elementsManager.captureImageButtonElement.addEventListener('click', this.captureImage.bind(this));
        this.elementsManager.trainNetworkButtonElement.addEventListener('click', this.trainModel.bind(this));
        this.elementsManager.removeModelButtonElement.addEventListener('click', this.removeModelFromStorage.bind(this));
        this.elementsManager.saveImagesButtonElement.addEventListener('click', () => { 
            this.dataContainer.save().then(() => {
                this.elementsManager.setMessageAndLog("Images saved");
            }); 
        });
        this.elementsManager.removeImagesButtonElement.addEventListener('click', this.removeAllImageTensors.bind(this));

        this.elementsManager.nextImageButtonElement.addEventListener('click', () => { this.cycleThroughImages(1); });
        this.elementsManager.previousImageButtonElement.addEventListener('click', () => { this.cycleThroughImages(-1); });

        this.elementsManager.removeImageButtonElement.addEventListener('click', this.removeCurrentImage.bind(this));
        this.elementsManager.featureSelectElement.addEventListener('input', this.changeImageStack.bind(this));

        this.elementsManager.startButtonElement.addEventListener('click', () => { this.setShouldMakePredictions(true); });
        this.elementsManager.stopButtonElement.addEventListener('click', () => { this.setShouldMakePredictions(false); });
        this.elementsManager.activationThresholdInputElement.addEventListener('input', (e) => { 
            this.activationThreshold = parseFloat(e.target.value);
            LocalStorage.save("activationThreshold", this.activationThreshold);
        });        
    }

    async loadShouldMakePredictions() {
        this.shouldMakePredictions = await LocalStorage.load("shouldMakePredictions");
    }

    setPredictionButtonsColors() {
        if (this.shouldMakePredictions) {
            this.elementsManager.startButtonElement.classList.remove("background-success");
            this.elementsManager.stopButtonElement.classList.add("background-alert");
        } else {
            this.elementsManager.startButtonElement.classList.add("background-success");
            this.elementsManager.stopButtonElement.classList.remove("background-alert");
        }
    }

    setShouldMakePredictions(shouldMakePredictions) {
        this.shouldMakePredictions = shouldMakePredictions;
        LocalStorage.save("shouldMakePredictions", this.shouldMakePredictions);
        this.setPredictionButtonsColors();
    }

    async loadActivationThreshold() {
        this.activationThreshold = await LocalStorage.load("activationThreshold");
        this.elementsManager.activationThresholdInputElement.value = this.activationThreshold;
    }

    loadDataContainer() {
        this.dataContainer.load().then(() => {
            if (this.dataContainer.xTrain != null) {
                let currentFeatureLabel = this.getCurrentFeatureLabel();
                this.loadImageWithIndexAndLabelIntoCanvas(this.currentImagePreviewIndex, currentFeatureLabel, this.elementsManager.imagePreviewCanvasElement);
                this.setExampleIndicator(currentFeatureLabel, this.elementsManager.numberOfExamplesElement);
                this.elementsManager.setMessageAndLog("Data container loaded");
            }
        });
    }

    removeCurrentImage() {
        let currentFeatureLabel = this.getCurrentFeatureLabel();
        this.dataContainer.removeTensorFromBatch(currentFeatureLabel, this.currentImagePreviewIndex);
        this.cycleThroughImages(-1);
        this.setExampleIndicator(currentFeatureLabel, this.elementsManager.numberOfExamplesElement);
        this.elementsManager.setMessageAndLog("Removing tensor successful");
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
            this.clearCanvas(this.elementsManager.imagePreviewCanvasElement);
            this.setExampleIndicator(this.currentFeatureLabel, this.elementsManager.numberOfExamplesElement)
            this.elementsManager.setMessageAndLog("Removing all tensors successful");
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

        this.elementsManager.setMessageAndLog("Image captured");
    }

    removeModelFromStorage() {
        this.model.removeFromStorage();
        this.elementsManager.setMessageAndLog("Model removed");
    }

    async trainModel() {
        let numberOfEpochs = parseInt(this.elementsManager.numberOfEpochsInputElement.value);
        this.elementsManager.setMessageAndLog("Train model");
        this.model.train(this.dataContainer.xTrain, this.dataContainer.yTrain, numberOfEpochs, this.updateTrainingProgress.bind(this)).then(() => {
            this.elementsManager.setMessageAndLog("Model trained");
        });
    }

    getCurrentFeatureLabel() {
        return this.elementsManager.featureSelectElement.options[this.elementsManager.featureSelectElement.selectedIndex].value;
    }

    setExampleIndicator(label, element) {
        let numberOfTensors = this.dataContainer.getNumberOfTensorsForLabel(label);

        if (numberOfTensors > 0) {
            let indicator = (this.currentImagePreviewIndex + 1) + "/" + numberOfTensors;
            element.innerHTML = indicator;
        } else {
            element.innerHTML = "";
        }
    }

    loadImageWithIndexAndLabelIntoCanvas(index, label, canvasElement) {
        let imageData = this.dataContainer.getTensorDataForYLabel(label, index);
        if (imageData != null) {
            tf.browser.toPixels(tf.tensor3d(imageData), canvasElement);
        } else {
            this.clearCanvas(canvasElement);
        }
    }

    clearCanvas(canvasElement) {
        canvasElement.getContext("2d").clearRect(0, 0, canvasElement.width, canvasElement.height);
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
        if (!this.shouldMakePredictions) return;

        let imageTensor = this.webcam.captureImageAndConvertToTensor();
        let resultTensor = tf.tidy(() => { return this.model.predict(imageTensor.expandDims(0)) });
        this.listPrediction(resultTensor);

        this.evaluatePredictionTensor(resultTensor);

        tf.dispose(imageTensor);
        tf.dispose(resultTensor);
    }

    listPrediction(prediction) {
        this.elementsManager.predictionList.innerHTML= "";
        let predictionArray = prediction.arraySync()[0];

        Object.keys(predictionArray).forEach(key => {
            let feature = this.features[key];
            let probability = predictionArray[key].toFixed(5);
            let listElement = document.createElement('li');
            listElement.appendChild(document.createTextNode(feature + ": " + probability));
            this.elementsManager.predictionList.appendChild(listElement);
        });
    }

    evaluatePredictionTensor(predictionTensor) {
        let maxPredictionValue = predictionTensor.max().dataSync()[0];
        if (maxPredictionValue < this.activationThreshold) return;

        let label = predictionTensor.argMax(1).dataSync()[0];

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
