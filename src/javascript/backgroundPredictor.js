class BackgroundPredictor {
    makePredictionEverySeconds(numberOfSeconds) {
        let milliseconds = numberOfSeconds * 1000;
        window.setInterval(this.makePrediction.bind(this), milliseconds)
    }

    makePrediction() {
        console.log("test");
    }
}

let backgroundPredictor = new BackgroundPredictor();
backgroundPredictor.makePredictionEverySeconds(1);