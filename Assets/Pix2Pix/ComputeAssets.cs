using UnityEngine;

namespace Pix2Pix
{
    class ComputeAssets : MonoBehaviour
    {
        #region Resources and accessors

        [SerializeField] ComputeShader _activation;
        [SerializeField] ComputeShader _concat;
        [SerializeField] ComputeShader _batchNorm;
        [SerializeField] ComputeShader _convolution;

        public static ComputeShader Activation  { get { return _instance._activation;  } }
        public static ComputeShader Concat      { get { return _instance._concat;      } } 
        public static ComputeShader BatchNorm   { get { return _instance._batchNorm;   } }
        public static ComputeShader Convolution { get { return _instance._convolution; } }

        #endregion

        #region Singleton-ish class implementation

        static ComputeAssets _instance;

        void OnEnable() { _instance = this; }
        void OnDisable() { _instance = null; }

        #endregion
    }
}
