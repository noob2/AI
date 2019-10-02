import 'tensorflow/tfjs'

let hiddenLayerConfig = {
  inputShape: [2],
  units: 30,
  activation: 'relu',
  useBias: true
}
let outputLayerConfig = { units: 1, useBias: true }
let compileConfig = { optimizer: tf.train.adam(), loss: 'meanSquaredError' }
let batchSize = 32
//----------------------------------------------------------------------------------------------------CONFIG
console.log(tf.getBackend())

let model = tf.sequential() //activation: sigmoid,relu, kernelInitializer: 'ones'
model.add(tf.layers.dense(hiddenLayerConfig))
model.add(tf.layers.dense(outputLayerConfig))
model.compile(compileConfig)
//{ loss: 'categoricalCrossentropy', optimizer: tf.train.sgd(0.2), metrics: ['acc'] }
//----------------------------------------------------------------------------------------------------PREPARE DATA
let tensorData = tf.tidy(() => {
  let shuffledData = EXAMPLE_DATA.output.map((output, i) => [
    EXAMPLE_DATA.input[i],
    EXAMPLE_DATA.input2[i],
    output
  ])
  tf.util.shuffle(shuffledData)
  let inputTensor = tf.tensor(shuffledData.map(el => [el[0], el[1]]))
  let outputTensor = tf.tensor(shuffledData.map(el => el[2]))
  let inputMax = inputTensor.max()
  let inputMin = inputTensor.min()
  let outputMax = outputTensor.max()
  let outputMin = outputTensor.min()
  let normalisedInput = inputTensor.sub(inputMin).div(inputMax.sub(inputMin))
  let normalisedOutput = outputTensor
    .sub(outputMin)
    .div(outputMax.sub(outputMin))
  return {
    input: normalisedInput,
    output: normalisedOutput,
    inputMax,
    inputMin,
    outputMax,
    outputMin
  }
})
//------------------------------------------------------------------------------------------------------TRAIN
let x_train = tensorData.input
let y_train = tensorData.output
let calculate = true
window.onload = () => {
  let button = document.getElementById('stop')
  button.addEventListener('click', () => {
    button.style.display = 'none'
    calculate = false
  })
}
;(async () => {
  let time = new Date().getTime()
  let i = 0
  while (calculate) {
    i++
    let res = await model.fit(x_train, y_train, {
      batchSize: batchSize,
      epochs: 200
    })
    document.getElementById('acc').innerHTML = res.history.loss[0].toFixed(9)
    let seconds = Math.round((new Date().getTime() - time) / 1000)
    document.getElementById('timer').innerHTML = seconds
    document.getElementById('timer2').innerHTML = i * 200
    if (seconds >= 200) break
  }
})().then(() => {
  //-------------------------------------------------------------------------------------------------------TEST
  let testTensor = tf.tensor(TEST_ARRAY)
  let normalisedTestTensor = testTensor
    .sub(tensorData.inputMin)
    .div(tensorData.inputMax.sub(tensorData.inputMin))
  const preds = model.predict(normalisedTestTensor)
  const unNormPreds = preds
    .mul(tensorData.outputMax.sub(tensorData.outputMin))
    .add(tensorData.outputMin)
  unNormPreds.dataSync().forEach((predictedOutput, i) => {
    let realInput = TEST_ARRAY[i]
    let realOutput = TEST_ARRAY[i][0] * TEST_ARRAY[i][1]
    var table = document.getElementById('result')
    var row = table.insertRow(1)
    var cell1 = row.insertCell(0)
    var cell2 = row.insertCell(1)
    var cell3 = row.insertCell(2)
    var cell4 = row.insertCell(3)
    cell1.innerHTML = realInput
    cell2.innerHTML = realOutput.toFixed(2)
    cell3.innerHTML = predictedOutput.toFixed(2)
    cell4.innerHTML =
      Math.abs(
        ((predictedOutput - realOutput) / predictedOutput) * 100
      ).toFixed(2) + '%'
  })
  document.getElementById('table').style.display = 'grid'
})
