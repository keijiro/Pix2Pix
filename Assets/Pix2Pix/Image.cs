// Image-tensor converter
// https://github.com/keijiro/Pix2Pix

using UnityEngine;

namespace Pix2Pix
{
    public static class Image
    {
        public static void ConvertToTensor(Texture source, Tensor tensor)
        {
            tensor.Reset(new Shape(source.height, source.width, 3));
            GpuBackend.InvokeImageToTensor(source, tensor);
        }

        public static Tensor ConvertToTensor(Texture source)
        {
            var tensor = new Tensor(new Shape(source.height, source.width, 3));
            ConvertToTensor(source, tensor);
            return tensor;
        }

        public static void ConvertFromTensor(Tensor source, RenderTexture destination)
        {
            GpuBackend.InvokeTensorToImage(source, destination);
        }

        public static Texture ConvertFromTensor(Tensor source)
        {
            var texture = new RenderTexture(source.Shape[1], source.Shape[0], 0);
            texture.enableRandomWrite = true;
            texture.Create();
            ConvertFromTensor(source, texture);
            return texture;
        }
    }
}
