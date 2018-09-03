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
            _weights = weights;
            _skip = new Stack<Tensor>();
        }

        public void Dispose()
        {
            Dispose(true);
            System.GC.SuppressFinalize(this);
        }

        ~Generator()
        {
            Dispose(false);
        }

        void Dispose(bool disposing)
        {
            if (disposing)
            {
                foreach (var tensor in _skip)
                    if (tensor != null) tensor.Dispose();

                _weights = null;
                _skip = null;
            }
        }

        #endregion

        #region Internal structure

        Dictionary<string, Tensor> _weights;
        Stack<Tensor> _skip;

        // Heuristic costs of each layer (total = 1,000)
        static readonly int[] _encoderCosts = { 15, 74, 74, 74, 40, 10, 6, 6 };
        static readonly int[] _decoderCosts = { 150, 150, 150, 150, 75, 20, 6, 0 };

        #endregion

        #region Generator coroutine

        public IEnumerator<int> Start(Tensor input, Tensor output)
        {
            {
                _skip.Push(Math.Conv2D(
                    input,
                    _weights["generator/encoder_1/conv2d/kernel"],
                    _weights["generator/encoder_1/conv2d/bias"]
                ));

                yield return _encoderCosts[0];
            }

            for (var i = 2; i <= 8; i++)
            {
                var scope = "generator/encoder_" + i;

                var kernel = _weights[scope + "/conv2d/kernel"];
                var bias   = _weights[scope + "/conv2d/bias"];
                var beta   = _weights[scope + "/batch_normalization/beta"];
                var gamma  = _weights[scope + "/batch_normalization/gamma"];

                var rect = Math.LeakyRelu(_skip.Peek(), 0.2f);
                var conv = Math.Conv2D(rect, kernel, bias);
                var norm = Math.BatchNorm(conv, gamma, beta);

                _skip.Push(norm);

                rect.Dispose();
                conv.Dispose();

                yield return _encoderCosts[i - 1];
            }

            var decoding = _skip.Pop();

            for (var i = 8; i >= 2; i--)
            {
                var scope = "generator/decoder_" + i;

                var kernel = _weights[scope + "/conv2d_transpose/kernel"];
                var bias   = _weights[scope + "/conv2d_transpose/bias"];
                var beta   = _weights[scope + "/batch_normalization/beta"];
                var gamma  = _weights[scope + "/batch_normalization/gamma"];

                if (i < 8)
                {
                    var prev = decoding;
                    var skip = _skip.Pop();

                    decoding = Math.Concat(prev, skip);

                    prev.Dispose();
                    skip.Dispose();
                }

                var rect = Math.Relu(decoding);
                var conv = Math.Deconv2D(rect, kernel, bias);
                var norm = Math.BatchNorm(conv, gamma, beta);

                rect.Dispose();
                conv.Dispose();
                decoding.Dispose();

                decoding = norm;

                yield return _decoderCosts[i - 1];
            }

            {
                var kernel = _weights["generator/decoder_1/conv2d_transpose/kernel"];
                var bias   = _weights["generator/decoder_1/conv2d_transpose/bias"];

                var skip = _skip.Pop();
                var join = Math.Concat(decoding, skip);
                var rect = Math.Relu(join);
                var conv = Math.Deconv2D(rect, kernel, bias);
                Math.Tanh(conv, output);

                decoding.Dispose();
                skip.Dispose();
                join.Dispose();
                rect.Dispose();
                conv.Dispose();

                yield return _decoderCosts[0];
            }
        }

        #endregion
    }
}
