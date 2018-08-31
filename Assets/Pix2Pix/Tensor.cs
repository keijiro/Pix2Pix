// Compute buffer backed tensor class
// https://github.com/keijiro/Pix2Pix

using System.Linq;

namespace Pix2Pix
{
    public sealed class Tensor : System.IDisposable
    {
        #region Public fields

        public int[] Shape;
        public UnityEngine.ComputeBuffer Buffer;

        #endregion

        #region Constructor and other common methods

        public Tensor(int[] shape, float[] data = null)
        {
            Shape = shape;

            var total = shape.Aggregate(1, (acc, x) => acc * x);
            Buffer = GpuHelper.AllocateBuffer(total);

            if (data != null)
            {
                UnityEngine.Debug.Assert(data.Length == total);
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
    }
}
