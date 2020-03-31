class ElementsManager {
    constructor() {
        this.videoElement = document.querySelector('#webcam-video');
        this.imagePreviewCanvasElement = document.querySelector('#image-preview-canvas');
        this.featureSelectElement = document.querySelector('#feature-select');
        this.numberOfExamplesElement = document.querySelector("#examples-span");
        this.numberOfEpochsInputElement = document.querySelector("#number-of-epochs-input");
        this.loadingBarElement = document.querySelector("#training-loading-bar");
        this.lossSpanElement = document.querySelector("#loss-span");
        this.activationThresholdInputElement = document.querySelector("#activation-threshold-input");
        this.predictionList = document.querySelector("#prediction-list");
        this.messageElement = document.querySelector("#message-board-message")

        this.initializeButtons();
    }

    initializeButtons() {
        this.captureImageButtonElement = document.querySelector("#capture-image-button");
        this.trainNetworkButtonElement = document.querySelector("#train-model-button");
        this.removeModelButtonElement = document.querySelector("#remove-model-button");
        this.saveImagesButtonElement = document.querySelector("#save-images-button");
        this.removeImagesButtonElement = document.querySelector("#remove-images-button");
        this.previousImageButtonElement = document.querySelector("#previous-image-button");
        this.nextImageButtonElement = document.querySelector("#next-image-button");
        this.removeImageButtonElement = document.querySelector("#remove-image-button")
        this.startButtonElement = document.querySelector("#start-button");
        this.stopButtonElement = document.querySelector("#stop-button");
    }

    getOptionsOfSelectElement(element) {
        let options = {};

        for (let index = 0; index < element.options.length; index++) {
            const option = element.options[index];
            options[option.value.toString()] = option.innerHTML;
        }

        return options;
    }

    setMessageAndLog(message) {
        console.log(message);
        this.messageElement.innerHTML = message;
    }
}

export default ElementsManager;