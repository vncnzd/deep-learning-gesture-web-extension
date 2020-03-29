import * as tf from '@tensorflow/tfjs';

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
      
            let oldYTrain = this.yTrain;
            this.yTrain = tf.keep(oldYTrain.concat(y, 0));
      
            oldXTrain.dispose();
            oldYTrain.dispose();
        }
    }

    load(storageKey = this.localStorageKey) {
        return browser.storage.local.get(storageKey).then((results) => {
            if (results[storageKey] != null) {
                let data = results[storageKey];
                this.xTrain = tf.tensor4d(Object.values(data.xTrain.data), data.xTrain.shape);
                this.yTrain = tf.tensor2d(Object.values(data.yTrain.data), data.yTrain.shape);
                console.log("Loading training data successful");
            } else {
                throw "No storage result for this key: " + storageKey;
            }
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

        return browser.storage.local.set({ [storageKey]: data }).then(() => {
            console.log("Saving training data successful");
        });
    }

    removeTensors(storageKey = this.localStorageKey) {
        return browser.storage.local.remove(storageKey).then(() => {
            this.xTrain = null;
            this.yTrain = null;
            console.log("Removing of stored images successful");
        });
    }

    getTensorData(index) {
        return this.xTrain.arraySync()[index];
    }

    removeTensorFromBatch(index) {
        let numberOfTensors = this.xTrain.shape[0];
        console.log(numberOfTensors);
        let indeces = Array.from(Array(numberOfTensors).keys())
        indeces.splice(index, 1)

        this.xTrain = tf.tidy(() => { return tf.gather(this.xTrain, indeces); });
        this.ytrain = tf.tidy(() => { return tf.gather(this.yTrain, indeces); });

        // maybe add save images

        console.log("Removing tensor successful");
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
        // TODO refactor this
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