// Compute buffer backed tensor class
// https://github.com/keijiro/Pix2Pix

using System.Linq;

namespace Pix2Pix
{
    // Tensor shape information struct
    public struct Shape
    {
        #region Public variables

        public readonly int Dim1, Dim2, Dim3, Dim4;

        #endregion

        #region Constructors

        public Shape(int dim1 = 0, int dim2 = 0, int dim3 = 0, int dim4 = 0)
        {
            Dim1 = dim1; Dim2 = dim2; Dim3 = dim3; Dim4 = dim4;
        }

        public Shape(int[] dimensions)
        {
            var l = dimensions.Length;
            Dim1 = l > 0 ? dimensions[0] : 0;
            Dim2 = l > 1 ? dimensions[1] : 0;
            Dim3 = l > 2 ? dimensions[2] : 0;
            Dim4 = l > 3 ? dimensions[3] : 0;
        }

        #endregion

        #region Public properties

        public int Order {
            get {
                if (Dim1 == 0) return 0;
                if (Dim2 == 0) return 1;
                if (Dim3 == 0) return 2;
                if (Dim4 == 0) return 3;
                return 4;
            }
        }

        public int this [int dim] {
            get {
                switch (dim)
                {
                    case 0: if (Dim1 > 0) return Dim1; else break;
                    case 1: if (Dim2 > 0) return Dim2; else break;
                    case 2: if (Dim3 > 0) return Dim3; else break;
                    case 3: if (Dim4 > 0) return Dim4; else break;
                }
                throw new System.IndexOutOfRangeException();
            }
        }

        public int ElementCount {
            get {
                return (Dim1 == 0 ? 1 : Dim1) *
                       (Dim2 == 0 ? 1 : Dim2) *
                       (Dim3 == 0 ? 1 : Dim3) *
                       (Dim4 == 0 ? 1 : Dim4);
            }
        }

        public override string ToString()
        {
            switch (Order)
            {
                case 1: return Dim1.ToString();
                case 2: return Dim1 + "x" + Dim2;
                case 3: return Dim1 + "x" + Dim2 + "x" + Dim3;
                case 4: return Dim1 + "x" + Dim2 + "x" + Dim3 + "x" + Dim4;
            }
            return "";
        }

        #endregion
    }

    // Compute buffer backed tensor class
    public sealed class Tensor : System.IDisposable
    {
        #region Public fields

        public Shape Shape;
        public UnityEngine.ComputeBuffer Buffer;

        #endregion

        #region Constructor and other common methods

        public Tensor()
        {
        }

        public Tensor(Shape shape, float[] data = null)
        {
            Shape = shape;

            var total = shape.ElementCount;
            if (total > 0) Buffer = GpuBackend.AllocateBuffer(total);

            if (data != null)
            {
                UnityEngine.Debug.Assert(data.Length == total);
                Buffer.SetData(data);
            }
        }

        public void Reset(Shape shape)
        {
            Shape = shape;

            var total = shape.ElementCount;

            if (Buffer != null && total != Buffer.count)
            {
                GpuBackend.ReleaseBuffer(Buffer);
                Buffer = null;
            }

            if (Buffer == null && total > 0)
                Buffer = GpuBackend.AllocateBuffer(total);
        }

        public override string ToString()
        {
            return "Tensor " + Shape;
        }

        #endregion

        #region IDisposable implementation

        public void Dispose()
        { 
            if (Buffer != null)
            {
                GpuBackend.ReleaseBuffer(Buffer);
                Buffer = null;
            }
        }

        ~Tensor()
        {
            if (Buffer != null)
                UnityEngine.Debug.LogError(
                    "Tensor (" + Shape + ") leaked. " +
                    "It must be explicitly disposed in the main thread."
                );
        }

        #endregion
    }
}
