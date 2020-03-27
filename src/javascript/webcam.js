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
        let self = this;

        return tf.tidy(function() {
            let imageTensor = tf.browser.fromPixels(self.videoElement);
            let quadraticImageTensor = self.sliceImageTensorToQuadratic(imageTensor);

            // The resizeBilinear function needs a shape of  shape [batch, height, width, inChannels].
            // That is why the dimension has to be expanded by one.
            let expandedImageTensor = quadraticImageTensor.expandDims(0);
            let resizedImageTensor = tf.image.resizeBilinear(expandedImageTensor, [self.imageTensorSize, self.imageTensorSize]);
            let normalizedImageTensor = tf.tidy(() => resizedImageTensor.toFloat().div(tf.scalar(255)));

            // tf.dispose(imageTensor);
            // tf.dispose(quadraticImageTensor);
            // tf.dispose(expandedImageTensor);
            // tf.dispose(resizedImageTensor);

            return tf.squeeze(normalizedImageTensor);
        });
    }

    sliceImageTensorToQuadratic(imageTensor) {
        let size = Math.min(imageTensor.shape[0], imageTensor.shape[1]);
		let startWidth = (imageTensor.shape[0] - size) / 2;
		let startHeight = (imageTensor.shape[1] - size) / 2;
        let resultTensor = imageTensor.slice([startWidth, startHeight, 0], [size, size, imageTensor.shape[2]]);

		return resultTensor;
    }
}

export default Webcam;
