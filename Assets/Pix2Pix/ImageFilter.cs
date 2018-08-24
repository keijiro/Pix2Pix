using UnityEngine;

namespace Pix2Pix
{
    static class ImageFilter
    {
        public static void Preprocess(Texture source, Tensor tensor)
        {
            var compute = ComputeAssets.Image;
            var kernel = compute.FindKernel("ImageToTensor");

            uint tgn_x, tgn_y, tgn_z;
            compute.GetKernelThreadGroupSizes(kernel, out tgn_x, out tgn_y, out tgn_z);

            var width  = source.width;
            var height = source.height;

            Debug.Assert(width  % tgn_x == 0);
            Debug.Assert(height % tgn_y == 0);

            Debug.Assert(tensor.Shape[0] == height);
            Debug.Assert(tensor.Shape[1] == width);
            Debug.Assert(tensor.Shape[2] == 3);

            compute.SetInts("Shape", tensor.Shape);
            compute.SetTexture(kernel, "InputImage", source);
            compute.SetBuffer(kernel, "OutputTensor", tensor.Buffer);
            compute.Dispatch(kernel, width / (int)tgn_x, height / (int)tgn_y, 1);
        }

        public static Tensor Preprocess(Texture source)
        {
            var tensor = new Tensor(new [] {source.height, source.width, 3});
            Preprocess(source, tensor);
            return tensor;
        }

        public static void Deprocess(Tensor source, RenderTexture destination)
        {
            var compute = ComputeAssets.Image;
            var kernel = compute.FindKernel("TensorToImage");

            uint tgn_x, tgn_y, tgn_z;
            compute.GetKernelThreadGroupSizes(kernel, out tgn_x, out tgn_y, out tgn_z);

            var width  = source.Shape[1];
            var height = source.Shape[0];

            Debug.Assert(width  % tgn_x == 0);
            Debug.Assert(height % tgn_y == 0);
            Debug.Assert(width  == destination.width);
            Debug.Assert(height == destination.height);
            Debug.Assert(destination.enableRandomWrite);

            compute.SetInts("Shape", source.Shape);
            compute.SetBuffer(kernel, "InputTensor", source.Buffer);
            compute.SetTexture(kernel, "OutputImage", destination);
            compute.Dispatch(kernel, width / (int)tgn_x, height / (int)tgn_y, 1);
        }

        public static Texture Deprocess(Tensor source)
        {
            var width  = source.Shape[1];
            var height = source.Shape[0];

            var texture = new RenderTexture(width, height, 0);
            texture.enableRandomWrite = true;
            texture.Create();

            Deprocess(source, texture);

            return texture;
        }
    }
}
