import * as tf from '@tensorflow/tfjs';
import LocalStorage from './localStorage';

class DataContainer {
    constructor() {
        this.localStorageKey = null;
        this.xTrain = null;
        this.yTrain = null;
    }

    add(x, y) {
        if (this.xTrain == null || this.yTrain == null) {
            this.xTrain = tf.keep(x);
            this.yTrain = tf.keep(y);
        } else {
            let oldXTrain = this.xTrain;
            this.xTrain = tf.keep(oldXTrain.concat(x, 0));
            oldXTrain.dispose();
      
            let oldYTrain = this.yTrain;
            this.yTrain = tf.keep(oldYTrain.concat(y, 0));
            oldYTrain.dispose();
        }
    }

    load(storageKey = this.localStorageKey) {
        return LocalStorage.load(storageKey).then((data) => {
            this.xTrain = tf.tensor4d(Object.values(data.xTrain.data), data.xTrain.shape);
            this.yTrain = tf.tensor2d(Object.values(data.yTrain.data), data.yTrain.shape);
        });
    }

    save(storageKey = this.localStorageKey) {
        let data = { 
            xTrain: { 
                data: this.xTrain.dataSync(), 
                shape: this.xTrain.shape 
            }, 
            yTrain: {
                data: this.yTrain.dataSync(),
                shape: this.yTrain.shape,
            } 
        };

        return LocalStorage.save(storageKey, data);
    }

    removeAllTensorsFromStorage(storageKey = this.localStorageKey) {
        return LocalStorage.delete(storageKey).then(() => {
            this.xTrain = null;
            this.yTrain = null;
        });
    }

    getTensorData(index) {
        return this.xTrain.arraySync()[index];
    }

    removeTensorFromBatch(label, labelIndex) {
        let tensorArray = this.yTrain.arraySync();
        let indexOfLabel = -1;

        for (let index = 0; index < tensorArray.length; index++) {
            let labelArray = tensorArray[index];

            if (labelArray.indexOf(1) == label) {
                indexOfLabel++;

                if (indexOfLabel == labelIndex) {
                    let numberOfTensors = this.xTrain.shape[0];

                    if (numberOfTensors != this.yTrain.shape[0]) {
                        console.log("yTrain and xTrain have a different number of tensors");
                    }

                    let indeces = Array.from(Array(numberOfTensors).keys());
                    indeces.splice(index, 1);

                    this.xTrain = tf.tidy(() => { return tf.gather(this.xTrain, indeces); });
                    this.yTrain = tf.tidy(() => { return tf.gather(this.yTrain, indeces); });
                    
                    return;
                }
            }
        }
    }

    getTensorDataForYLabel(label, index = null) {
        let indices = [];
        let tensorArray = this.yTrain.arraySync();

        for (let x = 0; x < tensorArray.length; x++) {
            let labelArray = tensorArray[x];

            if (labelArray.indexOf(1) == label) {
                indices.push(x);
            }
        }

        if (index != null) {
            return this.xTrain.gather(indices).arraySync()[index];
        } else {
            return this.xTrain.gather(indices).arraySync();
        }
    }

    getNumberOfTensorsForLabel(label) {
        if (this.yTrain == null) {
            return 0;
        }

        let indices = [];
        let tensorArray = this.yTrain.arraySync();

        for (let x = 0; x < tensorArray.length; x++) {
            let labelArray = tensorArray[x];

            if (labelArray.indexOf(1) == label) {
                indices.push(x);
            }
        }

        return indices.length;
    }
}

export default DataContainer;