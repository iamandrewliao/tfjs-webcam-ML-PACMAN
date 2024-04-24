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

/**
 * A dataset for webcam controls which allows the user to add example Tensors
 * for particular labels. This object will concat them into two large xs and ys.
 */
export class ControllerDataset {
  constructor(numClasses) {
    this.numClasses = numClasses;
  }

  /**
   * Adds an example to the controller dataset.
   * @param {Tensor} example A tensor representing the example. It can be an image,
   *     an activation, or any other type of Tensor.
   * @param {number} label The label of the example. Should be a number.
   */
  addExample(example, label) {
    // One-hot encode the label.
    const y = tf.tidy(
        () => tf.oneHot(tf.tensor1d([label]).toInt(), this.numClasses));
    console.log("One-hot encoding of adding. y:", y.array());

    if (this.xs == null) {
      // For the first example that gets added, keep example and y so that the
      // ControllerDataset owns the memory of the inputs. This makes sure that
      // if addExample() is called in a tf.tidy(), these Tensors will not get
      // disposed.
      this.xs = tf.keep(example);
      this.ys = tf.keep(y);
      // console.log("checking this.ys:", this.ys[0]);
    } else {
      const oldX = this.xs;
      this.xs = tf.keep(oldX.concat(example, 0));

      const oldY = this.ys;
      this.ys = tf.keep(oldY.concat(y, 0));

      oldX.dispose();
      oldY.dispose();
      y.dispose();
    }
  }

  clearDataset() {
    if (this.xs) {
      this.xs.dispose();
      this.ys.dispose();
      this.xs = null;
      this.ys = null;
    }
  }

  getOneHotEncodedArrayAtIndex(index) {
    return tf.tidy(() => {
      const ysArray = this.ys.arraySync();
      // console.log("One-hot encoded array:", ysArray);
      return ysArray[index].indexOf(1);
    });
  }


  // clear by label

  clearDatasetByLabel(labelToDelete) {
    if (!this.xs) {
        console.log("Dataset is empty. Nothing to delete.");
        return;
    }

    const indicesToDelete = [];
    const newXs = [];
    const newYs = [];

    // Iterate through each example and label
    for (let i = 0; i < this.ys.shape[0]; i++) {
        const label = this.getOneHotEncodedArrayAtIndex(i)
        console.log("label:", label);
        console.log("labelToDelete:", labelToDelete);

        // Check if the label matches the one to delete
        if (label !== labelToDelete) {
            // If not, keep the example and label
            newXs.push(this.xs.slice([i, 0], [1, this.xs.shape[1]]));
            newYs.push(this.ys.slice([i, 0], [1, this.ys.shape[1]]));
        } else {
            // If it matches, mark the index for deletion
            console.log("Index marked for deletion:", i);
            indicesToDelete.push(i);
        }
    }

    console.log("Indices to delete:", indicesToDelete);

    // Dispose of old tensors
    this.xs.dispose();
    this.ys.dispose();

    // If there are examples left after deletion
    if (newXs.length > 0) {
        console.log("Remaining examples after deletion:", newXs.length);
        // Concatenate remaining examples and labels
        this.xs = tf.keep(tf.concat(newXs, 0));
        this.ys = tf.keep(tf.concat(newYs, 0));
    } else {
        // If all examples were deleted, set to null
        console.log("All examples with label deleted.");
        this.xs = null;
        this.ys = null;
    }
  }
}

