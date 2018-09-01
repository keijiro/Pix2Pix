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
    public static class Generator
    {
        // Heuristic costs of each layer (total = 1,000)
        static readonly int[] _encoderCosts = { 15, 74, 74, 74, 40, 10, 6, 6 };
        static readonly int[] _decoderCosts = { 150, 150, 150, 150, 75, 20, 6, 0 };

        // Start the coroutine-like enumerator.
        public static IEnumerator<int> Start
            (Tensor input, Dictionary<string, Tensor> weights, Tensor output)
        {
            var layers = new Stack<Tensor>();

            {
                layers.Push(Math.Conv2D(
                    input,
                    weights["generator/encoder_1/conv2d/kernel"],
                    weights["generator/encoder_1/conv2d/bias"]
                ));

                yield return _encoderCosts[0];
            }

            for (var i = 2; i <= 8; i++)
            {
                var scope = "generator/encoder_" + i;

                var kernel = weights[scope + "/conv2d/kernel"];
                var bias   = weights[scope + "/conv2d/bias"];
                var beta   = weights[scope + "/batch_normalization/beta"];
                var gamma  = weights[scope + "/batch_normalization/gamma"];

                var rect = Math.LeakyRelu(layers.Peek(), 0.2f);
                var conv = Math.Conv2D(rect, kernel, bias);
                var norm = Math.BatchNorm(conv, gamma, beta);

                layers.Push(norm);

                rect.Dispose();
                conv.Dispose();

                yield return _encoderCosts[i - 1];
            }

            var decoding = layers.Pop();

            for (var i = 8; i >= 2; i--)
            {
                var scope = "generator/decoder_" + i;

                var kernel = weights[scope + "/conv2d_transpose/kernel"];
                var bias   = weights[scope + "/conv2d_transpose/bias"];
                var beta   = weights[scope + "/batch_normalization/beta"];
                var gamma  = weights[scope + "/batch_normalization/gamma"];

                if (i < 8)
                {
                    var prev = decoding;
                    var skip = layers.Pop();

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
                var kernel = weights["generator/decoder_1/conv2d_transpose/kernel"];
                var bias   = weights["generator/decoder_1/conv2d_transpose/bias"];

                var skip = layers.Pop();
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
    }
}
