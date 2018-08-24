using UnityEngine;

namespace Pix2Pix
{
    static class ImageFilter
    {
        public static Tensor Preprocess(Texture2D source)
        {
            var compute = ComputeAssets.Image;
            var kernel = compute.FindKernel("ImageToTensor");

            uint tgn_x, tgn_y, tgn_z;
            compute.GetKernelThreadGroupSizes(kernel, out tgn_x, out tgn_y, out tgn_z);

            var width  = source.width;
            var height = source.height;

            Debug.Assert(width  % tgn_x == 0);
            Debug.Assert(height % tgn_y == 0);

            var tensor = new Tensor(new [] {height, width, 3});

            compute.SetInts("Shape", tensor.Shape);
            compute.SetTexture(kernel, "InputImage", source);
            compute.SetBuffer(kernel, "OutputTensor", tensor.Buffer);
            compute.Dispatch(kernel, width / (int)tgn_x, height / (int)tgn_y, 1);

            return tensor;
        }

        public static Texture Deprocess(Tensor source)
        {
            var compute = ComputeAssets.Image;
            var kernel = compute.FindKernel("TensorToImage");

            uint tgn_x, tgn_y, tgn_z;
            compute.GetKernelThreadGroupSizes(kernel, out tgn_x, out tgn_y, out tgn_z);

            var width  = source.Shape[1];
            var height = source.Shape[0];

            Debug.Assert(width  % tgn_x == 0);
            Debug.Assert(height % tgn_y == 0);

            var texture = new RenderTexture(width, height, 0);
            texture.enableRandomWrite = true;
            texture.Create();

            compute.SetInts("Shape", source.Shape);
            compute.SetBuffer(kernel, "InputTensor", source.Buffer);
            compute.SetTexture(kernel, "OutputImage", texture);
            compute.Dispatch(kernel, width / (int)tgn_x, height / (int)tgn_y, 1);

            return texture;
        }
    }
}
