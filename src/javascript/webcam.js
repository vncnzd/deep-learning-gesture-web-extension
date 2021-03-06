import * as tf from '@tensorflow/tfjs';

class Webcam {
    constructor(videoElement, imageTensorSize) {
        this.videoElement = videoElement;
        this.imageTensorSize = imageTensorSize;
        this.setup(this.videoElement);
	}

    setup(videoElement) {
        navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
            videoElement.srcObject = stream;
        });
    }

    captureImageAndConvertToTensor() {
        return tf.tidy(() => {
            let imageTensor = tf.browser.fromPixels(this.videoElement);
            let quadraticImageTensor = this.sliceImageTensorToQuadratic(imageTensor);

            // The resizeBilinear function needs a shape of  shape [batch, height, width, inChannels].
            // That is why the dimension has to be expanded by one.
            let expandedImageTensor = quadraticImageTensor.expandDims(0);
            let resizedImageTensor = tf.image.resizeBilinear(expandedImageTensor, [this.imageTensorSize, this.imageTensorSize]);
            let normalizedImageTensor = tf.tidy(() => resizedImageTensor.toFloat().div(tf.scalar(255)));
            let squeezedImageTensor = tf.squeeze(normalizedImageTensor);

            tf.dispose(imageTensor);
            tf.dispose(quadraticImageTensor);
            tf.dispose(expandedImageTensor);
            tf.dispose(resizedImageTensor);
            tf.dispose(normalizedImageTensor);

            return squeezedImageTensor;
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
