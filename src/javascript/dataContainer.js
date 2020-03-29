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

    remove(storageKey = this.localStorageKey) {
        return browser.storage.local.remove(storageKey).then(() => {
            this.xTrain = null;
            this.yTrain = null;
            console.log("Removing of stored images successful");
        });
    }
}

export default DataContainer;