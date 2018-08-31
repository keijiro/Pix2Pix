using UnityEngine;
using UnityEngine.UI;
using System.IO;
using System.Collections.Generic;

namespace Pix2Pix
{
    public class SketchPad : MonoBehaviour
    {
        [SerializeField] string _weightFileName;
        [SerializeField] Renderer _sourceRenderer;
        [SerializeField] Renderer _resultRenderer;
        [SerializeField] Vector4 _drawArea;
        [SerializeField] Texture _defaultTexture;
        [SerializeField] Text _textDisplay;

        [SerializeField, HideInInspector] Shader _shader;

        RenderTexture _sourceTexture;
        RenderTexture _resultTexture;

        Material _lineMaterial;
        Mesh _lineMesh;

        Material _eraserMaterial;
        Mesh _eraserMesh;

        List<Vector3> _vertexList = new List<Vector3>(4);
        Vector3 _mouseHistory;

        Dictionary<string, Tensor> _weightTable;
        Tensor _sourceTensor;
        Tensor _resultTensor;

        Vector3 TransformMousePosition(Vector3 p)
        {
            var x = (p.x / Screen.width  - _drawArea.x) / (_drawArea.z - _drawArea.x);
            var y = (p.y / Screen.height - _drawArea.y) / (_drawArea.w - _drawArea.y);
            return new Vector3(x * 2 - 1, 1 - y * 2, 0);
        }

        void Start()
        {
            _sourceTexture = new RenderTexture(256, 256, 0);
            _resultTexture = new RenderTexture(256, 256, 0);

            _sourceTexture.filterMode = FilterMode.Point;
            _resultTexture.enableRandomWrite = true;

            _sourceTexture.Create();
            _resultTexture.Create();

            Graphics.Blit(_defaultTexture, _sourceTexture);

            _sourceRenderer.material.mainTexture = _sourceTexture;
            _resultRenderer.material.mainTexture = _resultTexture;

            _lineMaterial = new Material(_shader);
            _lineMaterial.color = Color.black;

            _lineMesh = new Mesh();
            _lineMesh.MarkDynamic();
            _lineMesh.vertices = new Vector3[2];
            _lineMesh.SetIndices(new[]{0, 1}, MeshTopology.Lines, 0);

            _eraserMaterial = new Material(_shader);
            _eraserMaterial.color = Color.white;

            _eraserMesh = new Mesh();
            _eraserMesh.MarkDynamic();
            _eraserMesh.vertices = new Vector3[4];
            _eraserMesh.SetIndices(new[]{0, 1, 2, 1, 3, 2}, MeshTopology.Triangles, 0);

            _weightTable = WeightReader.ReadFromFile(Path.Combine(Application.streamingAssetsPath, _weightFileName));
            _sourceTensor = new Tensor(new[]{256, 256, 3});
            _resultTensor = new Tensor(new[]{256, 256, 3});
        }

        void OnDestroy()
        {
            Destroy(_sourceTexture);
            Destroy(_resultTexture);

            Destroy(_lineMaterial);
            Destroy(_lineMesh);

            Destroy(_eraserMaterial);
            Destroy(_eraserMesh);

            WeightReader.DisposeTable(_weightTable);
            _sourceTensor.Dispose();
            _resultTensor.Dispose();
        }

        void Update()
        {
            UpdateSketch();
            UpdatePix2Pix();
        }

        void UpdateSketch()
        {
            var mousePosition = Input.mousePosition;

            var prevRT = RenderTexture.active;
            RenderTexture.active = _sourceTexture;

            if (Input.GetMouseButton(0))
            {
                if (Input.GetMouseButtonDown(0)) _mouseHistory = mousePosition;

                _vertexList.Clear();
                _vertexList.Add(TransformMousePosition(_mouseHistory));
                _vertexList.Add(TransformMousePosition(mousePosition));
                _lineMesh.SetVertices(_vertexList);

                _lineMaterial.SetPass(0);
                Graphics.DrawMeshNow(_lineMesh, Matrix4x4.identity);
            }
            else if (Input.GetMouseButton(1))
            {
                var p = TransformMousePosition(mousePosition);
                var d = 0.05f;

                _vertexList.Clear();
                _vertexList.Add(p + new Vector3(-d, -d, 0));
                _vertexList.Add(p + new Vector3(+d, -d, 0));
                _vertexList.Add(p + new Vector3(-d, +d, 0));
                _vertexList.Add(p + new Vector3(+d, +d, 0));
                _eraserMesh.SetVertices(_vertexList);

                _eraserMaterial.SetPass(0);
                Graphics.DrawMeshNow(_eraserMesh, Matrix4x4.identity);
            }

            RenderTexture.active = prevRT;
            _mouseHistory = mousePosition;
        }

        float _budget = 1200;
        float _spent;
        IEnumerator<int> _itr;

        readonly string [] _levels = {
            "N/A", "Poor", "Moderate", "Good", "Great", "Excellent"
        };

        void UpdatePix2Pix()
        {
            if (_itr == null)
            {
                Image.ConvertToTensor(_sourceTexture, _sourceTensor);
                _itr = Generator.Apply(_sourceTensor, _weightTable, _resultTensor);
            }

            while (_spent < _budget)
            {
                if (!_itr.MoveNext())
                {
                    Image.ConvertFromTensor(_resultTensor, _resultTexture);
                    _spent += _itr.Current;

                    Image.ConvertToTensor(_sourceTexture, _sourceTensor);
                    _itr = Generator.Apply(_sourceTensor, _weightTable, _resultTensor);
                }
                else
                {
                    _spent += _itr.Current;
                }
            }

            _spent -= _budget;

            _budget -= Mathf.Max(Time.deltaTime * 60 - 1.2f, 0) * 5;
            _budget = Mathf.Max(150, _budget);

            _textDisplay.text = "Refresh rate: " +
                (60 * Mathf.Min(1.0f, _budget / 1000)).ToString("0.0") +
                " Hz (" + _levels[(int)Mathf.Min(5, _budget / 100)] + ")";
        }
    }
}
