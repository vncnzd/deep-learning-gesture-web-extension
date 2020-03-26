import * as tf from '@tensorflow/tfjs';

class DataContainer {
    add(x, y) {
        if (this.xTrain == null) {
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
}

export default DataContainer;