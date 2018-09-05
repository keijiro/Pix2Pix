// Tensor math operations
// https://github.com/keijiro/Pix2Pix

namespace Pix2Pix
{
    public static class Math
    {
        // ReLU activation function

        public static void Relu(Tensor input, Tensor output)
        {
            output.Reset(input.Shape);
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
            output.Reset(input.Shape);
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
            output.Reset(input.Shape);
            GpuBackend.InvokeActivation("Tanh", input, 0, output);
        }

        public static Tensor Tanh(Tensor input)
        {
            var output = new Tensor(input.Shape);
            Tanh(input, output);
            return output;
        }

        // Tensor concatenation function

        static Shape ConcatShape(Tensor input1, Tensor input2)
        {
            return new Shape(input1.Shape[0], input1.Shape[1], input1.Shape[2] * 2);
        }

        public static void Concat(Tensor input1, Tensor input2, Tensor output)
        {
            output.Reset(ConcatShape(input1, input2));
            GpuBackend.InvokeConcat(input1, input2, output);
        }

        public static Tensor Concat(Tensor input1, Tensor input2)
        {
            var output = new Tensor(ConcatShape(input1, input2));
            Concat(input1, input2, output);
            return output;
        }

        // Batch normalization

        public static void BatchNorm(Tensor input, Tensor scale, Tensor offset, Tensor output)
        {
            output.Reset(input.Shape);
            GpuBackend.InvokeBatchNorm(input, scale, offset, output);
        }

        public static Tensor BatchNorm(Tensor input, Tensor scale, Tensor offset)
        {
            var output = new Tensor(input.Shape);
            GpuBackend.InvokeBatchNorm(input, scale, offset, output);
            return output;
        }

        // 2D convolution

        static Shape Conv2DShape(Tensor input, Tensor filter)
        {
            return new Shape(input.Shape[0] / 2, input.Shape[1] / 2, filter.Shape[3]);
        }

        public static void Conv2D(Tensor input, Tensor filter, Tensor bias, Tensor output)
        {
            output.Reset(Conv2DShape(input, filter));
            GpuBackend.InvokeConv2D(input, filter, bias, output);
        }

        public static Tensor Conv2D(Tensor input, Tensor filter, Tensor bias)
        {
            var output = new Tensor(Conv2DShape(input, filter));
            Conv2D(input, filter, bias, output);
            return output;
        }

        // 2D transposed convolution

        static Shape Deconv2DShape(Tensor input, Tensor filter)
        {
            return new Shape(input.Shape[0] * 2, input.Shape[1] * 2, filter.Shape[3]);
        }

        public static void Deconv2D(Tensor input, Tensor filter, Tensor bias, Tensor output)
        {
            output.Reset(Deconv2DShape(input, filter));
            GpuBackend.InvokeDeconv2D(input, filter, bias, output);
        }

        public static Tensor Deconv2D(Tensor input, Tensor filter, Tensor bias)
        {
            var output = new Tensor(Deconv2DShape(input, filter));
            Deconv2D(input, filter, bias, output);
            return output;
        }
    }
}
