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
            Buffer = new ComputeBuffer(total, sizeof(float));

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
            if (Buffer != null)
            {
                Buffer.Dispose();
                Buffer = null;
            }

            if (disposing) Shape = null;
        }

        #endregion

        #region Math operations

        public static Tensor Relu(Tensor input)
        {
            return GpuHelper.InvokeActivationKernel("Relu", input);
        }

        public static Tensor LeakyRelu(Tensor input, float alpha)
        {
            ComputeAssets.Activation.SetFloat("Alpha", alpha);
            return GpuHelper.InvokeActivationKernel("LeakyRelu", input);
        }

        public static Tensor Tanh(Tensor input)
        {
            return GpuHelper.InvokeActivationKernel("Tanh", input);
        }

        public static Tensor Concat(Tensor input1, Tensor input2)
        {
            var elements = input1.Shape[0] * input1.Shape[1];
            var kernel = elements < 512 ? "Concat64" : "Concat512";
            if (elements < 64) kernel = "Concat4";
            return GpuHelper.InvokeConcatKernel(kernel, input1, input2);
        }

        public static Tensor BatchNorm(Tensor input, Tensor scale, Tensor offset)
        {
            var channels = scale.Shape[0];
            var kernel = channels == 512 ? "BatchNorm512" : "BatchNorm64";
            return GpuHelper.InvokeBatchNormKernel(kernel, input, scale, offset);
        }

        public static Tensor Conv2D(Tensor input, Tensor filter, Tensor bias)
        {
            var outChannels = filter.Shape[3];
            var kernel = outChannels >= 512 ? "Conv2D_512_1" : "Conv2D_64_8";
            return GpuHelper.InvokeConvolutionKernel(GpuHelper.ConvolutionMode.Down, kernel, input, filter, bias);
        }

        public static Tensor Deconv2D(Tensor input, Tensor filter, Tensor bias)
        {
            var outChannels = filter.Shape[2];
            var kernel = outChannels >= 512 ? "TransConv2D_512_1" : "TransConv2D_64_8";
            if (outChannels == 3) kernel = "TransConv2D_3_128";
            return GpuHelper.InvokeConvolutionKernel(GpuHelper.ConvolutionMode.Up, kernel, input, filter, bias);
        }

        #endregion
    }
}
