// Pix2Pix generator model
// https://github.com/keijiro/Pix2Pix

// This class runs the network in a coroutine-like fashion. It doesn't use
// Unity's standard coroutine system but an ad-hoc enumerator to manage the
// progress and internal state. The enumerator returns a heuristic cost value
// for each step that can be used to estimate the GPU load.

using UnityEngine;
using System.Collections.Generic;

namespace Pix2Pix
{
    public class Generator : System.IDisposable
    {
        #region Constructor and IDisposable implementation

        public Generator(Dictionary<string, Tensor> weights)
        {
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
            while (_stack.Count > 0) _stack.Pop().Dispose();
        }

        ~Generator()
        {
            if (_stack.Count > 0)
                Debug.LogError("Generator instance must be explicitly disposed.");
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

        // Temporary tensor stack
        Stack<Tensor> _stack = new Stack<Tensor>();

        #endregion

        #region Generator coroutine

        public IEnumerator<int> Start(Tensor input, Tensor output)
        {
            var s = _stack;

            {
                var layer = _encoders[0];

                _stack.Push(Math.Conv2D(input, layer.Kernel, layer.Bias));

                yield return _encoderCosts[0];
            }

            for (var i = 1; i < 8; i++)
            {
                var layer = _encoders[i];

                s.Push(Math.LeakyRelu(s.Peek(), 0.2f));
                using (var t = s.Pop()) s.Push(Math.Conv2D(t, layer.Kernel, layer.Bias));
                using (var t = s.Pop()) s.Push(Math.BatchNorm(t, layer.Gamma, layer.Beta));

                yield return _encoderCosts[i];
            }

            for (var i = 7; i > 0; i--)
            {
                var layer = _decoders[i];

                if (i < 7)
                    using (Tensor t = s.Pop(), skip = s.Pop())
                        s.Push(Math.Concat(t, skip));

                using (var t = s.Pop()) s.Push(Math.Relu(t));
                using (var t = s.Pop()) s.Push(Math.Deconv2D(t, layer.Kernel, layer.Bias));
                using (var t = s.Pop()) s.Push(Math.BatchNorm(t, layer.Gamma, layer.Beta));

                yield return _decoderCosts[i];
            }

            {
                var layer = _decoders[0];

                using (Tensor t = s.Pop(), skip = s.Pop())
                    s.Push(Math.Concat(t, skip));

                using (var t = s.Pop()) s.Push(Math.Relu(t));
                using (var t = s.Pop()) s.Push(Math.Deconv2D(t, layer.Kernel, layer.Bias));
                using (var t = s.Pop()) Math.Tanh(t, output);

                yield return _decoderCosts[0];
            }
        }

        #endregion
    }
}
