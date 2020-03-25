import * as tf from '@tensorflow/tfjs';

class Webcam {
    constructor(videoElement) {
        this.videoElement = videoElement;
        this.imageTensorSize = 224;
        this.setup();
	}

    setup() {
        const self = this;

        navigator.mediaDevices.getUserMedia({ video: true }).then(function(stream) {
            self.videoElement.srcObject = stream;
            // self.videoElement.onloadeddata = function() {}
        });
    }

    captureImageAndGetTensor() {
        let imageTensor = tf.browser.fromPixels(this.videoElement);
        let quadraticImageTensor = this.sliceImageTensorToQuadratic(imageTensor);

        // The resizeBilinear function needs a shape of  shape [batch, height, width, inChannels].
        // That is why the dimension has to be expanded by one.
        let expandedImageTensor = tf.expandDims(quadraticImageTensor, 0);

        return tf.image.resizeBilinear(expandedImageTensor, [this.imageTensorSize, this.imageTensorSize]);
    }

    sliceImageTensorToQuadratic(imageTensor) {
        let size = Math.min(imageTensor.shape[0], imageTensor.shape[1]);
		let startWidth = (imageTensor.shape[0] - size) / 2;
		let startHeight = (imageTensor.shape[1] - size) / 2;
		let resultTensor = tf.slice(imageTensor, [startWidth, startHeight, 0], [size, size, imageTensor.shape[2]]);

		return resultTensor;
    }
}

export default Webcam;
