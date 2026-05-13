const tf = require('@tensorflow/tfjs-node');
async function test() {
  try {
    const model = await tf.loadLayersModel('file:///home/chielsy3992/furnsics/pwa_app/public/model/model.json');
    console.log('Model loaded successfully');
  } catch (err) {
    console.error('Error loading model:', err);
  }
}
test();
