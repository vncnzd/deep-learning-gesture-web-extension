import '../sass/style.scss';
import Webcam from './webcam.js';

let captureImageButton = document.querySelector("#capture-image-button");
let videoElement = document.querySelector('#webcam-video');
let webcam = new Webcam(videoElement);

captureImageButton.addEventListener('click', function() {
    console.log(webcam.captureImageAndGetTensor());
});

