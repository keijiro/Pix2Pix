using UnityEngine;
using System.Collections.Generic;

namespace Pix2Pix
{
    static class Generator
    {
        public static void Apply(Tensor input, Dictionary<string, Tensor> weights)
        {
            var layers = new Tensor[9];

            layers[1] = Tensor.EncodeConv2D(
                input,
                weights["generator/encoder_1/conv2d/kernel"],
                weights["generator/encoder_1/conv2d/bias"]
            );

            for (var i = 2; i < 9; i++)
            {
                var scope = "generator/encoder_" + i;

                layers[i] =
                    Tensor.BatchNorm(
                        Tensor.EncodeConv2D(
                            Tensor.LeakyRelu(layers[i - 1], 0.2f),
                            weights[scope + "/conv2d/kernel"],
                            weights[scope + "/conv2d/bias"]
                        ),
                        weights[scope + "/batch_normalization/gamma"],
                        weights[scope + "/batch_normalization/beta"]
                    );

                Debug.Log(layers[i]);
            }
        }
    }
}
