using UnityEngine;
using System.Collections.Generic;

namespace Pix2Pix
{
    static class Generator
    {
        static int[] _conv2DTimes = { 10, 45, 45, 45, 25, 5, 5, 5 };
        static int[] _deconv2DTimes = { 90, 90, 90, 90, 45, 15, 5, 1 };

        public static IEnumerator<int>
            Apply(Tensor input, Dictionary<string, Tensor> weights, Tensor output)
        {
            var layers = new Stack<Tensor>();

            layers.Push(Tensor.Conv2D(
                input,
                weights["generator/encoder_1/conv2d/kernel"],
                weights["generator/encoder_1/conv2d/bias"]
            ));

            yield return _conv2DTimes[0];

            for (var i = 2; i <= 8; i++)
            {
                var scope = "generator/encoder_" + i;

                var kernel = weights[scope + "/conv2d/kernel"];
                var bias   = weights[scope + "/conv2d/bias"];
                var beta   = weights[scope + "/batch_normalization/beta"];
                var gamma  = weights[scope + "/batch_normalization/gamma"];

                var rect = Tensor.LeakyRelu(layers.Peek(), 0.2f);
                var conv = Tensor.Conv2D(rect, kernel, bias);
                var norm = Tensor.BatchNorm(conv, gamma, beta);

                layers.Push(norm);

                rect.Dispose();
                conv.Dispose();

                yield return _conv2DTimes[i - 1];
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

                    decoding = Tensor.Concat(prev, skip);

                    prev.Dispose();
                    skip.Dispose();
                }

                var rect = Tensor.Relu(decoding);
                var conv = Tensor.Deconv2D(rect, kernel, bias);
                var norm = Tensor.BatchNorm(conv, gamma, beta);

                rect.Dispose();
                conv.Dispose();
                decoding.Dispose();

                decoding = norm;

                yield return _deconv2DTimes[i - 1];
            }

            {
                var kernel = weights["generator/decoder_1/conv2d_transpose/kernel"];
                var bias   = weights["generator/decoder_1/conv2d_transpose/bias"];

                var skip = layers.Pop();
                var join = Tensor.Concat(decoding, skip);
                var rect = Tensor.Relu(join);
                var conv = Tensor.Deconv2D(rect, kernel, bias);
                Tensor.Tanh(conv, output);

                decoding.Dispose();
                skip.Dispose();
                join.Dispose();
                rect.Dispose();
                conv.Dispose();

                yield return _deconv2DTimes[0];
            }
        }
    }
}
