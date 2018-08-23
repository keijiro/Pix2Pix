using UnityEngine;

namespace Pix2Pix
{
    class Pix2PixResources : MonoBehaviour
    {
        #region Resources and accessors

        [SerializeField] ComputeShader _compute;

        public static ComputeShader Compute { get { return _instance._compute; } }

        #endregion

        #region Singleton-ish class implementation

        static Pix2PixResources _instance;

        void OnEnable()
        {
            _instance = this;
        }

        void OnDisable()
        {
            _instance = null;
        }

        #endregion
    }
}
