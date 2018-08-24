using UnityEngine;

namespace Pix2Pix
{
    static class GpuHelper
    {
        public static Tensor InvokeActivationKernel(string name, Tensor input)
        {
            var compute = ComputeAssets.Activation;
            var kernel = compute.FindKernel(name);

            uint tgn_x, tgn_y, tgn_z;
            compute.GetKernelThreadGroupSizes(kernel, out tgn_x, out tgn_y, out tgn_z);
            Debug.Assert(tgn_y == 1 && tgn_z == 1);

            var length = input.Data.Length;
            Debug.Assert(length % tgn_x == 0);

            var buffer_input  = new ComputeBuffer(length, sizeof(float));
            var buffer_output = new ComputeBuffer(length, sizeof(float));

            buffer_input.SetData(input.Data);
            compute.SetBuffer(kernel, "Input", buffer_input);
            compute.SetBuffer(kernel, "Output", buffer_output);
            compute.Dispatch(kernel, length / (int)tgn_x, 1, 1);

            var output = new Tensor(input.Shape);
            buffer_output.GetData(output.Data);

            buffer_input .Dispose();
            buffer_output.Dispose();

            return output;
        }

        public static Tensor InvokeConcatKernel(string name, Tensor input1, Tensor input2)
        {
            var compute = ComputeAssets.Concat;
            var kernel = compute.FindKernel(name);

            uint tgn_x, tgn_y, tgn_z;
            compute.GetKernelThreadGroupSizes(kernel, out tgn_x, out tgn_y, out tgn_z);
            Debug.Assert(tgn_y == 1 && tgn_z == 1);

            var height = input1.Shape[0];
            var width  = input1.Shape[1];
            var channels1 = input1.Shape[2];
            var channels2 = input2.Shape[2];

            Debug.Assert(input2.Shape[0] == height);
            Debug.Assert(input2.Shape[1] == width);
            Debug.Assert(width * height % tgn_x == 0);

            var output = new Tensor(new [] {height, width, channels1 + channels2});

            var buffer_input1 = new ComputeBuffer(input1.Data.Length, sizeof(float));
            var buffer_input2 = new ComputeBuffer(input2.Data.Length, sizeof(float));
            var buffer_output = new ComputeBuffer(output.Data.Length, sizeof(float));

            buffer_input1.SetData(input1.Data);
            buffer_input2.SetData(input2.Data);

            compute.SetBuffer(kernel, "Input1", buffer_input1);
            compute.SetBuffer(kernel, "Input2", buffer_input2);
            compute.SetBuffer(kernel, "Output", buffer_output);

            compute.SetInts("Input1Shape", input1.Shape);
            compute.SetInts("Input2Shape", input2.Shape);

            compute.Dispatch(kernel, width * height / (int)tgn_x, 1, 1);
            buffer_output.GetData(output.Data);

            buffer_input1.Dispose();
            buffer_input2.Dispose();
            buffer_output.Dispose();

            return output;
        }

        public static Tensor InvokeBatchNormKernel(
            string name, Tensor input, Tensor scale, Tensor offset
        )
        {
            var compute = ComputeAssets.BatchNorm;
            var kernel = compute.FindKernel(name);

            uint tgn_x, tgn_y, tgn_z;
            compute.GetKernelThreadGroupSizes(kernel, out tgn_x, out tgn_y, out tgn_z);

            var length = input.Data.Length;
            var channels = input.Shape[2];

            Debug.Assert(channels % tgn_x == 0);
            Debug.Assert(channels == scale .Data.Length);
            Debug.Assert(channels == offset.Data.Length);

            var buffer_input  = new ComputeBuffer(length,   sizeof(float));
            var buffer_scale  = new ComputeBuffer(channels, sizeof(float));
            var buffer_offset = new ComputeBuffer(channels, sizeof(float));
            var buffer_output = new ComputeBuffer(length,   sizeof(float));

            buffer_input .SetData(input .Data);
            buffer_scale .SetData(scale .Data);
            buffer_offset.SetData(offset.Data);

            compute.SetInts("InputShape", input.Shape);

            compute.SetBuffer(kernel, "Input" , buffer_input );
            compute.SetBuffer(kernel, "Scale" , buffer_scale );
            compute.SetBuffer(kernel, "Offset", buffer_offset);
            compute.SetBuffer(kernel, "Output", buffer_output);

            compute.Dispatch(kernel, channels / (int)tgn_x, 1, 1);

            var output = new Tensor(input.Shape);
            buffer_output.GetData(output.Data);

            buffer_input .Dispose();
            buffer_scale .Dispose();
            buffer_offset.Dispose();
            buffer_output.Dispose();

            return output;
        }

        public enum ConvolutionMode { Down, Up }

        public static Tensor InvokeConvolutionKernel(
            ConvolutionMode mode, string name, Tensor input, Tensor filter, Tensor bias
        )
        {
            var compute = ComputeAssets.Convolution;
            var kernel = compute.FindKernel(name);

            uint tgn_x, tgn_y, tgn_z;
            compute.GetKernelThreadGroupSizes(kernel, out tgn_x, out tgn_y, out tgn_z);

            var trans = (mode == ConvolutionMode.Up);
            var outHeight = trans ? input.Shape[0] * 2 : input.Shape[0] / 2;
            var outWidth  = trans ? input.Shape[1] * 2 : input.Shape[1] / 2;
            var outChannels = filter.Shape[trans ? 2 : 3];

            Debug.Assert(outHeight   % tgn_z == 0);
            Debug.Assert(outWidth    % tgn_y == 0);
            Debug.Assert(outChannels % tgn_x == 0);

            var output = new Tensor(new [] {outHeight, outWidth, outChannels});

            var buffer_input  = new ComputeBuffer(input .Data.Length, sizeof(float));
            var buffer_filter = new ComputeBuffer(filter.Data.Length, sizeof(float));
            var buffer_bias   = new ComputeBuffer(bias  .Data.Length, sizeof(float));
            var buffer_output = new ComputeBuffer(output.Data.Length, sizeof(float));

            buffer_input .SetData(input .Data);
            buffer_filter.SetData(filter.Data);
            buffer_bias  .SetData(bias  .Data);

            compute.SetInts( "InputShape", input .Shape);
            compute.SetInts("FilterShape", filter.Shape);
            compute.SetInts("OutputShape", output.Shape);

            compute.SetBuffer(kernel, "Input" , buffer_input );
            compute.SetBuffer(kernel, "Filter", buffer_filter);
            compute.SetBuffer(kernel, "Bias"  , buffer_bias  );
            compute.SetBuffer(kernel, "Output", buffer_output);

            compute.Dispatch(kernel,
                outChannels / (int)tgn_x,
                outWidth    / (int)tgn_y,
                outHeight   / (int)tgn_z
            );

            buffer_output.GetData(output.Data);

            buffer_input .Dispose();
            buffer_filter.Dispose();
            buffer_bias  .Dispose();
            buffer_output.Dispose();

            return output;
        }
    }
}
