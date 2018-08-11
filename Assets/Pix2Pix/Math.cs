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
    }
}
