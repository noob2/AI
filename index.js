import * as tf from "@tensorflow/tfjs";
import * as data from "./data.js";

let hiddenLayerConfig = {
  inputShape: [2],
  units: 30,
  activation: "relu",
  useBias: true
};
let outputLayerConfig = { units: 1, useBias: true };
let compileConfig = { optimizer: tf.train.adam(), loss: "meanSquaredError" };
let batchSize = 128;
//----------------------------------------------------------------------------------------------------CONFIG

let model = tf.sequential(); //activation: sigmoid,relu, kernelInitializer: 'ones'
model.add(tf.layers.dense(hiddenLayerConfig));
model.add(tf.layers.dense(outputLayerConfig));
model.compile(compileConfig);
//{ loss: 'categoricalCrossentropy', optimizer: tf.train.sgd(0.2), metrics: ['acc'] }
//----------------------------------------------------------------------------------------------------PREPARE DATA
let tensorData = tf.tidy(() => {
  let shuffledData = data.EXAMPLE_DATA.output.map((output, i) => [
    data.EXAMPLE_DATA.input[i],
    data.EXAMPLE_DATA.input2[i],
    output
  ]); 
  tf.util.shuffle(shuffledData);
  let inputTensor = tf.tensor(shuffledData.map(el => [el[0], el[1]]));
  let outputTensor = tf.tensor(shuffledData.map(el => el[2]));
  let inputMax = inputTensor.max();
  let inputMin = inputTensor.min();
  let outputMax = outputTensor.max();
  let outputMin = outputTensor.min();
  let normalisedInput = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
  let normalisedOutput = outputTensor.sub(outputMin).div(outputMax.sub(outputMin));
  return {
    input: normalisedInput,
    output: normalisedOutput,
    inputMax,
    inputMin,
    outputMax,
    outputMin
  };
});
//------------------------------------------------------------------------------------------------------TRAIN
let x_train = tensorData.input;
let y_train = tensorData.output;
let calculate = true;
window.onload = () => {
  let button = document.getElementById("stop");
  button.addEventListener("click", () => {
    button.style.display = "none";
    calculate = false;
  });
};
(async () => {
  let time = new Date().getTime();
  while (calculate) {
    let res = await model.fit(x_train, y_train, {
      batchSize: batchSize,
      epochs: 50
    }); 
    document.getElementById("acc").innerHTML = Number.parseFloat(res.history.loss[0]).toExponential(2);
    let seconds = Math.round((new Date().getTime() - time) / 1000);
    document.getElementById("timer").innerHTML = seconds;
    if (seconds >= 25) break;
  }
})().then(() => {
  //-------------------------------------------------------------------------------------------------------TEST
  let testTensor = tf.tensor(data.TEST_ARRAY);
  let normalisedTestTensor = testTensor.sub(tensorData.inputMin).div(tensorData.inputMax.sub(tensorData.inputMin));
  const preds = model.predict(normalisedTestTensor);
  const unNormPreds = preds.mul(tensorData.outputMax.sub(tensorData.outputMin)).add(tensorData.outputMin);
 let totalDifference = 0;
  unNormPreds.dataSync().forEach((predictedOutput, i) => {
    let realInput = data.TEST_ARRAY[i];
    let realOutput = data.TEST_ARRAY[i][0] * data.TEST_ARRAY[i][1];
    var table = document.getElementById("result");
    var row = table.insertRow(1);
    var cell1 = row.insertCell(0);
    var cell2 = row.insertCell(1);
    var cell3 = row.insertCell(2);
    var cell4 = row.insertCell(3);
    cell1.innerHTML = realInput;
    cell2.innerHTML = realOutput.toFixed(2);
    cell3.innerHTML = predictedOutput.toFixed(2);
    let percentage = Math.abs(((predictedOutput - realOutput) / predictedOutput) * 100)
    totalDifference += percentage
    cell4.innerHTML =  percentage.toFixed(2)+ "%";
  });
  document.getElementById("acc").innerHTML += ' ---------- average error: '+(totalDifference/data.TEST_ARRAY.length).toFixed(2)+'%'
  document.getElementById("table").style.display = "grid";
});
