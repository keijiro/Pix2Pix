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
            _stack = new Stack<Tensor>();
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

        Dictionary<string, Tensor> _weights;
        Stack<Tensor> _stack;

        // Heuristic costs of each layer (total = 1,000)
        static readonly int[] _encoderCosts = { 15, 74, 74, 74, 40, 10, 6, 6 };
        static readonly int[] _decoderCosts = { 150, 150, 150, 150, 75, 20, 6, 0 };

        #endregion

        #region Generator coroutine

        public IEnumerator<int> Start(Tensor input, Tensor output)
        {
            var s = _stack;

            {
                var kernel = _weights["generator/encoder_1/conv2d/kernel"];
                var bias   = _weights["generator/encoder_1/conv2d/bias"];

                _stack.Push(Math.Conv2D(input, kernel, bias));

                yield return _encoderCosts[0];
            }

            for (var i = 2; i <= 8; i++)
            {
                var scope = "generator/encoder_" + i;

                var kernel = _weights[scope + "/conv2d/kernel"];
                var bias   = _weights[scope + "/conv2d/bias"];
                var beta   = _weights[scope + "/batch_normalization/beta"];
                var gamma  = _weights[scope + "/batch_normalization/gamma"];

                s.Push(Math.LeakyRelu(s.Peek(), 0.2f));
                using (var t = s.Pop()) s.Push(Math.Conv2D(t, kernel, bias));
                using (var t = s.Pop()) s.Push(Math.BatchNorm(t, gamma, beta));

                yield return _encoderCosts[i - 1];
            }

            for (var i = 8; i >= 2; i--)
            {
                var scope = "generator/decoder_" + i;

                var kernel = _weights[scope + "/conv2d_transpose/kernel"];
                var bias   = _weights[scope + "/conv2d_transpose/bias"];
                var beta   = _weights[scope + "/batch_normalization/beta"];
                var gamma  = _weights[scope + "/batch_normalization/gamma"];

                if (i < 8)
                {
                    using (Tensor t = s.Pop(), skip = s.Pop())
                        s.Push(Math.Concat(t, skip));
                }

                using (var t = s.Pop()) s.Push(Math.Relu(t));
                using (var t = s.Pop()) s.Push(Math.Deconv2D(t, kernel, bias));
                using (var t = s.Pop()) s.Push(Math.BatchNorm(t, gamma, beta));

                yield return _decoderCosts[i - 1];
            }

            {
                var kernel = _weights["generator/decoder_1/conv2d_transpose/kernel"];
                var bias   = _weights["generator/decoder_1/conv2d_transpose/bias"];

                using (Tensor t = s.Pop(), skip = s.Pop())
                    s.Push(Math.Concat(t, skip));

                using (var t = s.Pop()) s.Push(Math.Relu(t));
                using (var t = s.Pop()) s.Push(Math.Deconv2D(t, kernel, bias));
                using (var t = s.Pop()) Math.Tanh(t, output);

                yield return _decoderCosts[0];
            }
        }

        #endregion
    }
}
