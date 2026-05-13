const tf = require('@tensorflow/tfjs');

const origCall = tf.layers.GlobalAveragePooling2D.prototype.call;
tf.layers.GlobalAveragePooling2D.prototype.call = function(inputs, kwargs) {
    const out = origCall.call(this, inputs, kwargs);
    console.log("Patched call! out shape:", out.shape);
    if (this.keepDims || this.keepdims) {
         return tf.reshape(out, [out.shape[0], 1, 1, out.shape[1]]);
    }
    return out;
};

const origShape = tf.layers.GlobalAveragePooling2D.prototype.computeOutputShape;
tf.layers.GlobalAveragePooling2D.prototype.computeOutputShape = function(inputShape) {
    const shape = origShape.call(this, inputShape);
    if (this.keepDims || this.keepdims) {
         return [shape[0], 1, 1, shape[1]];
    }
    return shape;
};

const l = tf.layers.globalAveragePooling2d({keepDims: true});
const t = tf.ones([2, 5, 5, 3]);
const out = l.apply(t);
console.log("Final shape:", out.shape);
