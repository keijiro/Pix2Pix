using UnityEngine;
using System.Collections.Generic;

namespace Pix2Pix
{
    static class Generator
    {
        public static void Apply(Tensor input, Dictionary<string, Tensor> weights)
        {
            var filter = weights["generator/encoder_1/conv2d/kernel"];
            var bias = weights["generator/encoder_1/conv2d/bias"];

            var conv = Tensor.EncodeConv2D(input, filter, bias);
        }
    }
}
