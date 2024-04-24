/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '@tensorflow/tfjs';
import * as tfd from '@tensorflow/tfjs-data';

import {ControllerDataset} from './controller_dataset';
import * as ui from './ui';
import { totals } from './ui.js';

// The number of classes we want to predict. In this example, we will be
// predicting 4 classes for up, down, left, and right.
const NUM_CLASSES = 4;

// A webcam iterator that generates Tensors from the images from the webcam.
let webcam;

// The dataset object where we will store activations.
const controllerDataset = new ControllerDataset(NUM_CLASSES);

let truncatedMobileNet;
let model;

const CONTROLS = ['up', 'down', 'left', 'right'];

// Loads mobilenet and returns a model that returns the internal activation
// we'll use as input to our classifier model.
async function loadTruncatedMobileNet() {
  const mobilenet = await tf.loadLayersModel(
      'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');

  // Return a model that outputs an internal activation.
  const layer = mobilenet.getLayer('conv_pw_13_relu');
  return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
}

// When the UI buttons are pressed, read a frame from the webcam and associate
// it with the class label given by the button. up, down, left, right are
// labels 0, 1, 2, 3 respectively.
ui.setExampleHandler(async label => {
  let img = await getImage();

  controllerDataset.addExample(truncatedMobileNet.predict(img), label);

  // Draw the preview thumbnail.
  ui.drawThumb(img, label);
  img.dispose();
})

/**
 * Sets up and trains the classifier.
 */
async function train() {
  if (controllerDataset.xs == null) {
    throw new Error('Add some examples before training!');
  }

  // Creates a 2-layer fully connected model. By creating a separate model,
  // rather than adding layers to the mobilenet model, we "freeze" the weights
  // of the mobilenet model, and only train weights from the new model.
  model = tf.sequential({
    layers: [
      // Flattens the input to a vector so we can use it in a dense layer. While
      // technically a layer, this only performs a reshape (and has no training
      // parameters).
      tf.layers.flatten(
          {inputShape: truncatedMobileNet.outputs[0].shape.slice(1)}),
      // Layer 1.
      tf.layers.dense({
        units: ui.getDenseUnits(),
        activation: 'relu',
        kernelInitializer: 'varianceScaling',
        useBias: true
      }),
      // Layer 2. The number of units of the last layer should correspond
      // to the number of classes we want to predict.
      tf.layers.dense({
        units: NUM_CLASSES,
        kernelInitializer: 'varianceScaling',
        useBias: false,
        activation: 'softmax'
      })
    ]
  });

  // Creates the optimizers which drives training of the model.
  const optimizer = tf.train.adam(ui.getLearningRate());
  // We use categoricalCrossentropy which is the loss function we use for
  // categorical classification which measures the error between our predicted
  // probability distribution over classes (probability that an input is of each
  // class), versus the label (100% probability in the true class)>
  model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});

  // We parameterize batch size as a fraction of the entire dataset because the
  // number of examples that are collected depends on how many examples the user
  // collects. This allows us to have a flexible batch size.
  const batchSize =
      Math.floor(controllerDataset.xs.shape[0] * ui.getBatchSizeFraction());
  if (!(batchSize > 0)) {
    throw new Error(
        `Batch size is 0 or NaN. Please choose a non-zero fraction.`);
  }

  // Train the model! Model.fit() will shuffle xs & ys so we don't have to.
  model.fit(controllerDataset.xs, controllerDataset.ys, {
    batchSize,
    epochs: ui.getEpochs(),
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        ui.trainStatus('Loss: ' + logs.loss.toFixed(5));
      },
      onTrainEnd: () => {
        // Set the text of the button to "Complete" when training is done.
        ui.trainStatus('Training completed!');
        // ui.trainStatus('Training complete! Loss: ' + logs.loss.toFixed(5));
        // if want loss: additional variable
      }
    }
  });
}

let isPredicting = false;

async function predict() {
  ui.isPredicting();
  while (isPredicting) {
    // Capture the frame from the webcam.
    const img = await getImage();

    // Make a prediction through mobilenet, getting the internal activation of
    // the mobilenet model, i.e., "embeddings" of the input images.
    const embeddings = truncatedMobileNet.predict(img);

    // Make a prediction through our newly-trained model using the embeddings
    // from mobilenet as input.
    const predictions = model.predict(embeddings);

    // Returns the index with the maximum probability. This number corresponds
    // to the class the model thinks is the most probable given the input.
    const predictedClass = predictions.as1D().argMax();
    const classId = (await predictedClass.data())[0];
    img.dispose();

    ui.predictClass(classId);
    await tf.nextFrame();
  }
  ui.donePredicting();
}

/**
 * Captures a frame from the webcam and normalizes it between -1 and 1.
 * Returns a batched image (1-element batch) of shape [1, w, h, c].
 */
async function getImage() {
  const img = await webcam.capture();
  const processedImg =
      tf.tidy(() => img.expandDims(0).toFloat().div(127).sub(1));
  img.dispose();
  return processedImg;
}

document.getElementById('train').addEventListener('click', async () => {
  ui.trainStatus('Training...');
  await tf.nextFrame();
  await tf.nextFrame();
  isPredicting = false;
  train();
});
document.getElementById('predict').addEventListener('click', () => {
  ui.startPacman();
  isPredicting = true;
  predict();
});

document.getElementById('clear').addEventListener('click', () => {
  controllerDataset.clearDataset();
  // controllerDataset.clearDatasetByLabel(0);
  resetInterface();
});
document.getElementById('up-clear').addEventListener('click', () => {
  resetInterfaceByLabel(0);
  controllerDataset.clearDatasetByLabel(0);
});
document.getElementById('down-clear').addEventListener('click', () => {
  resetInterfaceByLabel(1);
  controllerDataset.clearDatasetByLabel(1);
});
document.getElementById('left-clear').addEventListener('click', () => {
  resetInterfaceByLabel(2);
  controllerDataset.clearDatasetByLabel(2);
});
document.getElementById('right-clear').addEventListener('click', () => {
  resetInterfaceByLabel(3);
  controllerDataset.clearDatasetByLabel(3);
});


async function resetInterface() {
  // reset the number of instances in each class to 0
  for (let i = 0; i < CONTROLS.length; ++i) {
    const className = CONTROLS[i];
    const total = document.getElementById(className + '-total');
    total.innerText = 0;

    // reset ui.totals to [0,0,0,0]
    // Tried totals = [0,0,0,0] outside of the loop, did not work. I wonder why.
    totals[i] = 0;
  }
  // TODO: maybe reset the appearance of 4 buttons
}

async function resetInterfaceByLabel(label) {
  // reset the number of instances in the class specified by the label to 0
  const className = CONTROLS[label];
  const total = document.getElementById(className + '-total');
  total.innerText = 0;

  // reset ui.totals[label] to 0
  totals[label] = 0;
}

//functions for testing section
let testTotal = 0;
let test_imgs = {};

async function predictImage() {  // helper function
  isPredicting = true;
  // Capture the frame from the webcam.
  const img = await getImage();

  // Make a prediction through mobilenet, getting the internal activation of
  // the mobilenet model, i.e., "embeddings" of the input images.
  const embeddings = truncatedMobileNet.predict(img);

  // Make a prediction through our newly-trained model using the embeddings
  // from mobilenet as input.
  const predictions = model.predict(embeddings);

  // Returns the index with the maximum probability. This number corresponds
  // to the class the model thinks is the most probable given the input.
  const predictedClass = predictions.as1D().argMax();
  const classId = (await predictedClass.data())[0];
  img.dispose();
  isPredicting=false;
  return classId
}

async function testImage() {  // predicts image, shows prediction, and updates count
  // update number of test images collected (testTotal)
  testTotal+=1;
  console.log('testTotal: ', testTotal)
  let classId = await predictImage();
  test_imgs[testTotal] = {};  //initialize dictionary
  test_imgs[testTotal]['prediction'] = classIdtoName(classId); // update dictionary
  // show model's prediction
  displayPrediction(classId);
  const testTotalElement = document.getElementById('test-total');
  testTotalElement.textContent = testTotal.toString() + " test images collected";
}

function classIdtoName(classId){  // helper
  let className;
  if (classId == 0){
    className = "UP";
  } else if (classId == 1){
    className = "DOWN";
  } else if (classId == 2){
    className = "LEFT";
  } else {
    className = "RIGHT";
  }
  return className;
}

function displayPrediction(classId) {
  const predictionResultElement = document.getElementById('prediction-result');
  let className = classIdtoName(classId);
  predictionResultElement.textContent = className;
}

function summaryStats() {
  let up_count = 0;
  let down_count = 0;
  let left_count = 0;
  let right_count = 0;
  let up_wrong = 0;
  let down_wrong = 0;
  let left_wrong = 0;
  let right_wrong = 0;
  for(let img in test_imgs){
    if(test_imgs[img]['trueLabel'] == 'UP'){
      up_count += 1;
      if(test_imgs[img]['prediction'] != test_imgs[img]['trueLabel']){
        up_wrong += 1;
      }
    }
    if(test_imgs[img]['trueLabel'] == 'DOWN'){
      down_count += 1;
      if(test_imgs[img]['prediction'] != test_imgs[img]['trueLabel']){
        down_wrong += 1;
      }
    }
    if(test_imgs[img]['trueLabel'] == 'LEFT'){
      left_count += 1;
      if(test_imgs[img]['prediction'] != test_imgs[img]['trueLabel']){
        left_wrong += 1;
      }
    }
    if(test_imgs[img]['trueLabel'] == 'RIGHT'){
      right_count += 1;
      if(test_imgs[img]['prediction'] != test_imgs[img]['trueLabel']){
        right_wrong += 1;
      }
    }
  }
  let up_acc = (up_count-up_wrong)/up_count;
  let down_acc = (down_count-down_wrong)/down_count;
  let left_acc = (left_count-left_wrong)/left_count;
  let right_acc = (right_count-right_wrong)/right_count;
  let up_acc_html = document.getElementById("up_acc");
  let down_acc_html = document.getElementById("down_acc");
  let left_acc_html = document.getElementById("left_acc");
  let right_acc_html = document.getElementById("right_acc");
  up_acc_html.textContent = 'UP: ' + String(up_acc.toFixed(2));
  down_acc_html.textContent = 'DOWN: ' + String(down_acc.toFixed(2));
  left_acc_html.textContent = 'LEFT: ' + String(left_acc.toFixed(2));
  right_acc_html.textContent = 'RIGHT: ' + String(right_acc.toFixed(2));
}

function displayStats() {
  let ssp = document.getElementById("summaryStatsPanel")
  if (ssp.style.display === 'none') {
    ssp.style.display = 'block';
    summaryStats();
  } else {
    ssp.style.display = 'none';
  }
}

function recordTrueLabel(trueLabel){
  test_imgs[testTotal]['trueLabel'] = trueLabel;
  console.log(test_imgs);
}

async function init() {
  try {
    webcam = await tfd.webcam(document.getElementById('webcam'));
    test_webcam = await tfd.webcam(document.getElementById('test-webcam'));
  } catch (e) {
    console.log(e);
    document.getElementById('no-webcam').style.display = 'block';
  }
  truncatedMobileNet = await loadTruncatedMobileNet();

  ui.init();

  // Warm up the model. This uploads weights to the GPU and compiles the WebGL
  // programs so the first time we collect data from the webcam it will be
  // quick.
  const screenShot = await webcam.capture();
  truncatedMobileNet.predict(screenShot.expandDims(0));
  screenShot.dispose();
  document.getElementById("summaryStatsPanel").style.display ="none";
  // testing section
  document.getElementById('test-capture').addEventListener('click', testImage); //webcam capture
  document.getElementById('up-button').addEventListener('click', () => recordTrueLabel('UP'));
  document.getElementById('down-button').addEventListener('click', () => recordTrueLabel('DOWN'));
  document.getElementById('left-button').addEventListener('click', () => recordTrueLabel('LEFT'));
  document.getElementById('right-button').addEventListener('click', () => recordTrueLabel('RIGHT'));
  document.getElementById('summaryStats').addEventListener('click', displayStats);
}

// Initialize the application.
init();
