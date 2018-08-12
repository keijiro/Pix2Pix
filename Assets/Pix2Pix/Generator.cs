using UnityEngine;
using System.Collections.Generic;

namespace Pix2Pix
{
    static class Generator
    {
        public static Tensor Apply(Tensor input, Dictionary<string, Tensor> weights)
        {
            var layers = new Stack<Tensor>();

            layers.Push(Tensor.Conv2D(
                input,
                weights["generator/encoder_1/conv2d/kernel"],
                weights["generator/encoder_1/conv2d/bias"]
            ));

            for (var i = 2; i <= 8; i++)
            {
                var scope = "generator/encoder_" + i;

                layers.Push(
                    Tensor.BatchNorm(
                        Tensor.Conv2D(
                            Tensor.LeakyRelu(layers.Peek(), 0.2f),
                            weights[scope + "/conv2d/kernel"],
                            weights[scope + "/conv2d/bias"]
                        ),
                        weights[scope + "/batch_normalization/gamma"],
                        weights[scope + "/batch_normalization/beta"]
                    )
                );
            }

            var decoding = layers.Pop();

            for (var i = 8; i >= 2; i--)
            {
                var scope = "generator/decoder_" + i;

                if (i < 8) decoding = Tensor.Concat(decoding, layers.Pop());

                decoding =
                    Tensor.BatchNorm(
                        Tensor.Deconv2D(
                            Tensor.Relu(decoding),
                            weights[scope + "/conv2d_transpose/kernel"],
                            weights[scope + "/conv2d_transpose/bias"]
                        ),
                        weights[scope + "/batch_normalization/gamma"],
                        weights[scope + "/batch_normalization/beta"]
                    );
            }

            return Tensor.Tanh(
                Tensor.Deconv2D(
                    Tensor.Relu(Tensor.Concat(decoding, layers.Pop())),
                    weights["generator/decoder_1/conv2d_transpose/kernel"],
                    weights["generator/decoder_1/conv2d_transpose/bias"]
                )
            );
        }
    }
}
