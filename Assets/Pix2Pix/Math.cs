using UnityEngine;
using System.Linq;

namespace Pix2Pix
{
    public sealed class Tensor : System.IDisposable
    {
        #region Public fields

        public int[] Shape;
        public ComputeBuffer Buffer;

        #endregion

        #region Constructor and other common methods

        public Tensor(int[] shape, float[] data = null)
        {
            Shape = shape;

            var total = shape.Aggregate(1, (acc, x) => acc * x);
            Buffer = GpuHelper.AllocateBuffer(total);

            if (data != null)
            {
                Debug.Assert(data.Length == total);
                Buffer.SetData(data);
            }
        }

        public override string ToString()
        {
            return "Tensor " +
                string.Join("x", Shape.Select(x => x.ToString()).ToArray());
        }

        #endregion

        #region IDisposable implementation

        public void Dispose()
        { 
            Dispose(true);
            System.GC.SuppressFinalize(this);           
        }

        ~Tensor()
        {
            Dispose(false);
        }

        void Dispose(bool disposing)
        {
            if (disposing)
            {
                Shape = null;

                if (Buffer != null)
                {
                    GpuHelper.ReleaseBuffer(Buffer);
                    Buffer = null;
                }
            }
        }

        #endregion

        #region Math operations

        public static Tensor Relu(Tensor input)
        {
            var output = new Tensor(input.Shape);
            GpuHelper.InvokeActivationKernel("Relu", input, output);
            return output;
        }

        public static Tensor LeakyRelu(Tensor input, float alpha)
        {
            var output = new Tensor(input.Shape);
            ComputeAssets.Activation.SetFloat("Alpha", alpha);
            GpuHelper.InvokeActivationKernel("LeakyRelu", input, output);
            return output;
        }

        public static Tensor Tanh(Tensor input)
        {
            var output = new Tensor(input.Shape);
            GpuHelper.InvokeActivationKernel("Tanh", input, output);
            return output;
        }

        public static void Tanh(Tensor input, Tensor output)
        {
            GpuHelper.InvokeActivationKernel("Tanh", input, output);
        }

        public static Tensor Concat(Tensor input1, Tensor input2)
        {
            return GpuHelper.InvokeConcatKernel(input1, input2);
        }

        public static Tensor BatchNorm(Tensor input, Tensor scale, Tensor offset)
        {
            return GpuHelper.InvokeBatchNormKernel(input, scale, offset);
        }

        public static Tensor Conv2D(Tensor input, Tensor filter, Tensor bias)
        {
            var kernel = "Conv2D_" + filter.Shape[3];
            return GpuHelper.InvokeConvolutionKernel(GpuHelper.ConvolutionMode.Down, kernel, input, filter, bias);
        }

        public static Tensor Deconv2D(Tensor input, Tensor filter, Tensor bias)
        {
            var channels = filter.Shape[3];
            var kernel = "TransConv2D_" + (channels == 3 ? "final" : channels.ToString());
            return GpuHelper.InvokeConvolutionKernel(GpuHelper.ConvolutionMode.Up, kernel, input, filter, bias);
        }

        #endregion
    }
}
