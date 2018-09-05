// Pix2Pix generator model
// https://github.com/keijiro/Pix2Pix

// This class runs the network in a time-sliced fashion. The Step() method
// returns a heuristic cost value for each step that can be used to estimate
// the GPU load.

using UnityEngine;
using System.Collections.Generic;

namespace Pix2Pix
{
    public class Generator : System.IDisposable
    {
        #region Constructor and IDisposable implementation

        public Generator(Dictionary<string, Tensor> weights)
        {
            // Initialize tensor objects with empty tensors.
            for (var i = 0; i < 8; i ++) _skip[i] = new Tensor();
            _temp1 = new Tensor();
            _temp2 = new Tensor();

            // Load all layer weights from the given dictionary.
            _encoders[0] = new Layer {
                Kernel = weights["generator/encoder_1/conv2d/kernel"],
                Bias   = weights["generator/encoder_1/conv2d/bias"]
            };

            _decoders[0] = new Layer {
                Kernel = weights["generator/decoder_1/conv2d_transpose/kernel"],
                Bias   = weights["generator/decoder_1/conv2d_transpose/bias"]
            };

            for (var i = 1; i < 8; i++)
            {
                var scope = "generator/encoder_" + (i + 1);

                _encoders[i] = new Layer {
                    Kernel = weights[scope + "/conv2d/kernel"],
                    Bias   = weights[scope + "/conv2d/bias"],
                    Beta   = weights[scope + "/batch_normalization/beta"],
                    Gamma  = weights[scope + "/batch_normalization/gamma"]
                };

                scope = "generator/decoder_" + (i + 1);

                _decoders[i] = new Layer {
                    Kernel = weights[scope + "/conv2d_transpose/kernel"],
                    Bias   = weights[scope + "/conv2d_transpose/bias"],
                    Beta   = weights[scope + "/batch_normalization/beta"],
                    Gamma  = weights[scope + "/batch_normalization/gamma"]
                };
            }
        }

        public void Dispose()
        {
            foreach (var t in _skip) t.Dispose();
            _temp1.Dispose();
            _temp2.Dispose();
        }

        #endregion

        #region Internal structure

        // Heuristic costs of each layer (total = 1,000)
        static readonly int[] _encoderCosts = { 15, 74, 74, 74, 40, 10, 6, 6 };
        static readonly int[] _decoderCosts = { 150, 150, 150, 150, 75, 20, 6, 0 };

        // Encoder/decoder layers
        struct Layer { public Tensor Kernel, Bias, Beta, Gamma; }
        Layer[] _encoders = new Layer[8];
        Layer[] _decoders = new Layer[8];

        // Temporary tensors
        Tensor[] _skip = new Tensor[8];
        Tensor _temp1;
        Tensor _temp2;

        int _progress;

        #endregion

        #region Public properties and methods

        public bool Running { get { return _progress > 0 && _progress < 16; } }

        public void Start(Texture input)
        {
            Image.ConvertToTensor(input, _temp1);
            _progress = 0;
        }

        public void GetResult(RenderTexture output)
        {
            Image.ConvertFromTensor(_temp1, output);
        }

        public int Step()
        {
            if (_progress == 0)
            {
                var layer = _encoders[0];
                Math.Conv2D(_temp1, layer.Kernel, layer.Bias, _skip[0]);
            }
            else if (_progress < 8)
            {
                var layer = _encoders[_progress];
                Math.LeakyRelu(_skip[_progress - 1], 0.2f, _temp1);
                Math.Conv2D(_temp1, layer.Kernel, layer.Bias, _temp2);
                Math.BatchNorm(_temp2, layer.Gamma, layer.Beta, _skip[_progress]);
            }
            else if (_progress == 8)
            {
                var layer = _decoders[7];
                Math.Relu(_skip[7], _temp1);
                Math.Deconv2D(_temp1, layer.Kernel, layer.Bias, _temp2);
                Math.BatchNorm(_temp2, layer.Gamma, layer.Beta, _temp1);
            }
            else if (_progress < 15)
            {
                var i = 15 - _progress;
                var layer = _decoders[i];
                Math.Concat(_temp1, _skip[i], _temp2);
                Math.Relu(_temp2, _temp1);
                Math.Deconv2D(_temp1, layer.Kernel, layer.Bias, _temp2);
                Math.BatchNorm(_temp2, layer.Gamma, layer.Beta, _temp1);
            }
            else
            {
                var layer = _decoders[0];
                Math.Concat(_temp1, _skip[0], _temp2);
                Math.Relu(_temp2, _temp1);
                Math.Deconv2D(_temp1, layer.Kernel, layer.Bias, _temp2);
                Math.Tanh(_temp2, _temp1);
            }

            var cost = (_progress < 8) ?
                _encoderCosts[_progress] : _decoderCosts[15 - _progress];

            _progress++;

            return cost;
        }

        #endregion
    }
}
