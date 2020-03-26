import * as tf from '@tensorflow/tfjs';
import '../sass/style.scss';
import Webcam from './webcam.js';

class App {
    constructor() {
        this.captureImageButton = document.querySelector("#capture-image-button");
        this.videoElement = document.querySelector('#webcam-video');
        this.webcam = new Webcam(this.videoElement);

        this.addEventListeners();
    }

    addEventListeners() {
        let self = this;

        this.captureImageButton.addEventListener('click', function() {
            let imageTensor = self.webcam.captureImageAndGetTensor();
            let expandedImageTensor = tf.expandDims(imageTensor, 0);

            if (self.imageTensors == undefined) {
                self.imageTensors = expandedImageTensor
            } else {
                self.imageTensors = tf.concat([self.imageTensors, expandedImageTensor], 0);
            }

            console.log(self.imageTensors);
        });
    }
}

let app = new App();

