// Tensor math operations
// https://github.com/keijiro/Pix2Pix

namespace Pix2Pix
{
    public static class Math
    {
        // ReLU activation function

        public static void Relu(Tensor input, Tensor output)
        {
            GpuBackend.InvokeActivation("Relu", input, 0, output);
        }

        public static Tensor Relu(Tensor input)
        {
            var output = new Tensor(input.Shape);
            Relu(input, output);
            return output;
        }

        // Leaky ReLU activation function

        public static void LeakyRelu(Tensor input, float alpha, Tensor output)
        {
            GpuBackend.InvokeActivation("LeakyRelu", input, alpha, output);
        }

        public static Tensor LeakyRelu(Tensor input, float alpha)
        {
            var output = new Tensor(input.Shape);
            LeakyRelu(input, alpha, output);
            return output;
        }

        // Tanh activation function

        public static void Tanh(Tensor input, Tensor output)
        {
            GpuBackend.InvokeActivation("Tanh", input, 0, output);
        }

        public static Tensor Tanh(Tensor input)
        {
            var output = new Tensor(input.Shape);
            Tanh(input, output);
            return output;
        }

        // Tensor concatenation function

        public static void Concat(Tensor input1, Tensor input2, Tensor output)
        {
            GpuBackend.InvokeConcat(input1, input2, output);
        }

        public static Tensor Concat(Tensor input1, Tensor input2)
        {
            var output = new Tensor(new[]{
                input1.Shape[0],
                input1.Shape[1],
                input1.Shape[2] * 2
            });
            Concat(input1, input2, output);
            return output;
        }

        // Batch normalization

        public static void BatchNorm(Tensor input, Tensor scale, Tensor offset, Tensor output)
        {
            GpuBackend.InvokeBatchNorm(input, scale, offset, output);
        }

        public static Tensor BatchNorm(Tensor input, Tensor scale, Tensor offset)
        {
            var output = new Tensor(input.Shape);
            GpuBackend.InvokeBatchNorm(input, scale, offset, output);
            return output;
        }

        // 2D convolution

        public static void Conv2D(Tensor input, Tensor filter, Tensor bias, Tensor output)
        {
            GpuBackend.InvokeConv2D(input, filter, bias, output);
        }

        public static Tensor Conv2D(Tensor input, Tensor filter, Tensor bias)
        {
            var output = new Tensor(new[]{
                input.Shape[0] / 2,
                input.Shape[1] / 2,
                filter.Shape[3]
            });
            Conv2D(input, filter, bias, output);
            return output;
        }

        // 2D transposed convolution

        public static void Deconv2D(Tensor input, Tensor filter, Tensor bias, Tensor output)
        {
            GpuBackend.InvokeDeconv2D(input, filter, bias, output);
        }

        public static Tensor Deconv2D(Tensor input, Tensor filter, Tensor bias)
        {
            var output = new Tensor(new[]{
                input.Shape[0] * 2,
                input.Shape[1] * 2,
                filter.Shape[3]
            });
            Deconv2D(input, filter, bias, output);
            return output;
        }
    }
}
