// Pix2Pix sketch pad demo
// https://github.com/keijiro/Pix2Pix

using UnityEngine;
using UnityEngine.EventSystems;
using System.Collections.Generic;
using UI = UnityEngine.UI;

public class SketchPad : MonoBehaviour
{
    #region Editable attributes

    [SerializeField] string _weightFileName = "";
    [SerializeField] Texture _defaultTexture = null;

    [SerializeField] UI.RawImage _sourceUI = null;
    [SerializeField] UI.RawImage _resultUI = null;
    [SerializeField] UI.Text _textUI = null;

    [SerializeField, HideInInspector] Shader _drawShader = null;

    #endregion

    #region Internal objects

    RenderTexture _sourceTexture;
    RenderTexture _resultTexture;

    Material _lineMaterial;
    Material _eraserMaterial;

    List<Vector3> _vertexList = new List<Vector3>(4);
    Mesh _lineMesh;
    Mesh _eraserMesh;

    #endregion

    #region Drawing UI implementation

    public void OnDrag(BaseEventData baseData)
    {
        var data = (PointerEventData)baseData;
        data.Use();

        var area = data.pointerDrag.GetComponent<RectTransform>();
        var p0 = area.InverseTransformPoint(data.position - data.delta);
        var p1 = area.InverseTransformPoint(data.position);

        var scale = new Vector3(2 / area.rect.width, -2 / area.rect.height, 0);
        p0 = Vector3.Scale(p0, scale);
        p1 = Vector3.Scale(p1, scale);

        var eraser = (data.button == PointerEventData.InputButton.Right);
        eraser |= Input.GetKey(KeyCode.LeftShift) || Input.GetKey(KeyCode.RightShift);

        DrawSegment(p0, p1, eraser);
    }

    void DrawSegment(Vector3 p0, Vector3 p1, bool isEraser)
    {
        var prevRT = RenderTexture.active;
        RenderTexture.active = _sourceTexture;

        if (!isEraser)
        {
            _vertexList.Clear();
            _vertexList.Add(p0);
            _vertexList.Add(p1);
            _lineMesh.SetVertices(_vertexList);

            _lineMaterial.SetPass(0);
            Graphics.DrawMeshNow(_lineMesh, Matrix4x4.identity);
        }
        else
        {
            const float d = 0.05f;

            _vertexList.Clear();
            _vertexList.Add(p0 + new Vector3(-d, -d, 0));
            _vertexList.Add(p0 + new Vector3(+d, -d, 0));
            _vertexList.Add(p0 + new Vector3(-d, +d, 0));
            _vertexList.Add(p0 + new Vector3(+d, +d, 0));
            _eraserMesh.SetVertices(_vertexList);

            _eraserMaterial.SetPass(0);
            Graphics.DrawMeshNow(_eraserMesh, Matrix4x4.identity);
        }

        RenderTexture.active = prevRT;
    }

    #endregion

    #region Pix2Pix implementation

    Dictionary<string, Pix2Pix.Tensor> _weightTable;
    Pix2Pix.Generator _generator;

    float _budget = 100;
    float _budgetAdjust = 10;

    readonly string [] _performanceLabels = {
        "N/A", "Poor", "Moderate", "Good", "Great", "Excellent"
    };

    void InitializePix2Pix()
    {
        var filePath = System.IO.Path.Combine(Application.streamingAssetsPath, _weightFileName);
        _weightTable = Pix2Pix.WeightReader.ReadFromFile(filePath);
        _generator = new Pix2Pix.Generator(_weightTable);
    }

    void FinalizePix2Pix()
    {
        _generator.Dispose();
        Pix2Pix.WeightReader.DisposeTable(_weightTable);
    }

    void UpdatePix2Pix()
    {
        // Advance the Pix2Pix inference until the current budget runs out.
        for (var cost = 0.0f; cost < _budget;)
        {
            if (!_generator.Running) _generator.Start(_sourceTexture);

            cost += _generator.Step();

            if (!_generator.Running) _generator.GetResult(_resultTexture);
        }

        Pix2Pix.GpuBackend.ExecuteAndClearCommandBuffer();

        // Review the budget depending on the current frame time.
        _budget -= (Time.deltaTime * 60 - 1.25f) * _budgetAdjust;
        _budget = Mathf.Clamp(_budget, 150, 1200);

        _budgetAdjust = Mathf.Max(_budgetAdjust - 0.05f, 0.5f);

        // Update the text display.
        var rate = 60 * _budget / 1000;

        var perf = (_budgetAdjust < 1) ?
            _performanceLabels[(int)Mathf.Min(5, _budget / 100)] :
            "Measuring GPU performance...";

        _textUI.text =
            string.Format("Pix2Pix refresh rate: {0:F1} Hz ({1})", rate, perf);
    }

    #endregion

    #region MonoBehaviour implementation

    void Start()
    {
        // Texture/image initialization
        _sourceTexture = new RenderTexture(256, 256, 0);
        _resultTexture = new RenderTexture(256, 256, 0);

        _sourceTexture.filterMode = FilterMode.Point;
        _resultTexture.enableRandomWrite = true;

        _sourceTexture.Create();
        _resultTexture.Create();

        Graphics.Blit(_defaultTexture, _sourceTexture);

        _sourceUI.texture = _sourceTexture;
        _resultUI.texture = _resultTexture;

        // Draw object initialization
        _lineMaterial = new Material(_drawShader);
        _eraserMaterial = new Material(_drawShader);

        _lineMaterial.color = Color.black;
        _eraserMaterial.color = Color.white;

        _lineMesh = new Mesh();
        _lineMesh.MarkDynamic();
        _lineMesh.vertices = new Vector3[2];
        _lineMesh.SetIndices(new[]{0, 1}, MeshTopology.Lines, 0);

        _eraserMesh = new Mesh();
        _eraserMesh.MarkDynamic();
        _eraserMesh.vertices = new Vector3[4];
        _eraserMesh.SetIndices(new[]{0, 1, 2, 1, 3, 2}, MeshTopology.Triangles, 0);

        // Pix2Pix initialization
        InitializePix2Pix();
    }

    void OnDestroy()
    {
        Destroy(_sourceTexture);
        Destroy(_resultTexture);

        Destroy(_lineMaterial);
        Destroy(_eraserMaterial);

        Destroy(_lineMesh);
        Destroy(_eraserMesh);

        FinalizePix2Pix();
    }

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.C))
            Graphics.Blit(_defaultTexture, _sourceTexture);

        UpdatePix2Pix();
    }

    #endregion
}
