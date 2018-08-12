using System.Linq;

namespace Pix2Pix
{
    public struct Tensor
    {
        public int[] Shape;
        public float[] Data;

        public Tensor(int[] shape, float[] data)
        {
            Shape = shape;
            Data = data;
        }

        public Tensor(int[] shape)
        {
            Shape = shape;
            Data = new float[shape.Aggregate(1, (acc, x) => acc * x)];
        }

        public override string ToString()
        {
            return "Tensor " +
                string.Join("x", Shape.Select(x => x.ToString()).ToArray());
        }

        public float Get(int i0)
        {
            UnityEngine.Debug.Assert(Shape.Length == 1);
            if (i0 < 0 || i0 >= Shape[0]) return 0;
            return Data[i0];
        }

        public float Get(int i0, int i1)
        {
            UnityEngine.Debug.Assert(Shape.Length == 2);
            if (i0 < 0 || i0 >= Shape[0]) return 0;
            if (i1 < 0 || i1 >= Shape[1]) return 0;
            return Data[i0 * Shape[1] + i1];
        }

        public float Get(int i0, int i1, int i2)
        {
            UnityEngine.Debug.Assert(Shape.Length == 3);
            if (i0 < 0 || i0 >= Shape[0]) return 0;
            if (i1 < 0 || i1 >= Shape[1]) return 0;
            if (i2 < 0 || i2 >= Shape[2]) return 0;
            return Data[(i0 * Shape[1] + i1) * Shape[2] + i2];
        }

        public float Get(int i0, int i1, int i2, int i3)
        {
            UnityEngine.Debug.Assert(Shape.Length == 4);
            if (i0 < 0 || i0 >= Shape[0]) return 0;
            if (i1 < 0 || i1 >= Shape[1]) return 0;
            if (i2 < 0 || i2 >= Shape[2]) return 0;
            if (i3 < 0 || i3 >= Shape[3]) return 0;
            return Data[((i0 * Shape[1] + i1) * Shape[2] + i2) * Shape[3] + i3];
        }

        public void Set(int i0, float value)
        {
            UnityEngine.Debug.Assert(Shape.Length == 1);
            if (i0 < 0 || i0 >= Shape[0]) return;
            Data[i0] = value;
        }

        public void Set(int i0, int i1, float value)
        {
            UnityEngine.Debug.Assert(Shape.Length == 2);
            if (i0 < 0 || i0 >= Shape[0]) return;
            if (i1 < 0 || i1 >= Shape[1]) return;
            Data[i0 * Shape[1] + i1] = value;
        }

        public void Set(int i0, int i1, int i2, float value)
        {
            UnityEngine.Debug.Assert(Shape.Length == 3);
            if (i0 < 0 || i0 >= Shape[0]) return;
            if (i1 < 0 || i1 >= Shape[1]) return;
            if (i2 < 0 || i2 >= Shape[2]) return;
            Data[(i0 * Shape[1] + i1) * Shape[2] + i2] = value;
        }

        public void Set(int i0, int i1, int i2, int i3, float value)
        {
            UnityEngine.Debug.Assert(Shape.Length == 4);
            if (i0 < 0 || i0 >= Shape[0]) return;
            if (i1 < 0 || i1 >= Shape[1]) return;
            if (i2 < 0 || i2 >= Shape[2]) return;
            if (i3 < 0 || i3 >= Shape[3]) return;
            Data[((i0 * Shape[1] + i1) * Shape[2] + i2) * Shape[3] + i3] = value;
        }

        public static Tensor LeakyRelu(Tensor input, float alpha)
        {
            var data = new float[input.Data.Length];
            for (var i = 0; i < data.Length; i++)
            {
                var v = input.Data[i];
                data[i] = v < 0 ? v * alpha : v;
            }
            return new Tensor(input.Shape, data);
        }

        public static Tensor BatchNorm(Tensor input, Tensor scale, Tensor offset)
        {
            UnityEngine.Debug.Assert(input.Shape.Length == 3);

            var output = new Tensor(input.Shape);
            var epsilon = 1e-5f;

            for (var ch = 0; ch < input.Shape[2]; ch++)
            {
                var mean = 0.0f;
                for (var y = 0; y < input.Shape[1]; y++)
                    for (var x = 0; x < input.Shape[0]; x++)
                        mean += input.Get(x, y, ch);
                mean /= input.Shape[0] + input.Shape[1];

                var variance = 0.0f;
                for (var y = 0; y < input.Shape[1]; y++)
                    for (var x = 0; x < input.Shape[0]; x++)
                        variance += MathUtil.Square(input.Get(x, y, ch) - mean);
                variance /= input.Shape[0] + input.Shape[1];

                var offs = offset.Get(ch);
                var sc = scale.Get(ch);

                for (var y = 0; y < input.Shape[1]; y++)
                    for (var x = 0; x < input.Shape[0]; x++)
                        output.Set(x, y, ch,
                            offs + (input.Get(x, y, ch) - mean) * sc /
                            UnityEngine.Mathf.Sqrt(variance + epsilon)
                        );
            }

            return output;
        }

        public static Tensor EncodeConv2D(Tensor input, Tensor filter, Tensor bias)
        {
            const int stride = 2;

            var inWidth = input.Shape[0];
            var inHeight = input.Shape[1];
            var inChannels = input.Shape[2];

            var outWidth = inWidth / stride;
            var outHeight = inHeight / stride;
            var outChannels = filter.Shape[3];

            var filterWidth = filter.Shape[0];
            var filterHeight = filter.Shape[1];

            var output = new Tensor(new [] {outWidth, outHeight, outChannels});

            for (var oc = 0; oc < outChannels; oc++)
            {
                for (var oy = 0; oy < outHeight; oy++)
                {
                    var ymin = oy * stride - filterHeight / 2;

                    for (var ox = 0; ox < outWidth; ox++)
                    {
                        var xmin = ox * stride - filterWidth / 2;
                        var prod = 0.0f;

                        for (var fy = 0; fy < filterHeight; fy++)
                        {
                            for (var fx = 0; fx < filterWidth; fx++)
                            {
                                for (var ic = 0; ic < inChannels; ic++)
                                {
                                    var pixel = input.Get(xmin + fx, ymin + fy, ic);
                                    var weight = filter.Get(fx, fy, ic, oc);
                                    prod += pixel * weight;
                                }
                            }
                        }

                        output.Set(ox, oy, oc, prod + bias.Get(oc));
                    }
                }
            }

            return output;
        }
    }
}
