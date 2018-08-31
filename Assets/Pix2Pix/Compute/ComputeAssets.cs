// Compute shader asset aggregation
// https://github.com/keijiro/Pix2Pix

using UnityEngine;

namespace Pix2Pix
{
    //[CreateAssetMenu(fileName = "ComputeAssets", menuName = "Pix2Pix/Compute Assets")]
    sealed class ComputeAssets : ScriptableObject
    {
        public ComputeShader Activation = null;
        public ComputeShader BatchNorm = null;
        public ComputeShader Concat = null;
        public ComputeShader Convolution = null;
        public ComputeShader Image = null;
        public ComputeShader Setup = null;
    }
}
