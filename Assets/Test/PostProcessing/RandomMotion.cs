// Gradient noise-based random motion
// https://github.com/keijiro/Pix2Pix

using UnityEngine;
using Unity.Mathematics;

class RandomMotion : MonoBehaviour
{
    [SerializeField] uint _seed = 1234;
    [SerializeField] float _frequency = 1;
    [SerializeField] float3 _positionAmplitude = 1;
    [SerializeField] float3 _rotationAmplitude = 90;
    [SerializeField] float _scaleAmplitude = 0;

    float3 _originalPosition;
    quaternion _originalRotation;
    float _originalScale;

    float3 _positionNoise;
    float3 _rotationNoise;
    float _scaleNoise;

    void Start()
    {
        _originalPosition = transform.localPosition;
        _originalRotation = transform.localRotation;
        _originalScale = transform.localScale.x;

        var r = new Unity.Mathematics.Random(_seed);

        _positionNoise = r.NextFloat3() * 100;
        _rotationNoise = r.NextFloat3() * 100;
        _scaleNoise = r.NextFloat() * 100;
    }

    void Update()
    {
        var t = Time.time * _frequency;

        var px = noise.snoise(math.float2(_positionNoise.x, t));
        var py = noise.snoise(math.float2(_positionNoise.y, t));
        var pz = noise.snoise(math.float2(_positionNoise.z, t));
        var p = math.float3(px, py, pz) * _positionAmplitude;

        var rx = noise.snoise(math.float2(_rotationNoise.x, t));
        var ry = noise.snoise(math.float2(_rotationNoise.y, t));
        var rz = noise.snoise(math.float2(_rotationNoise.z, t));
        var r = math.float3(rx, ry, rz) * math.radians(_rotationAmplitude);

        var s = noise.snoise(math.float2(_scaleNoise, t)) * _scaleAmplitude;

        transform.localPosition = _originalPosition + p;
        transform.localRotation = math.mul(quaternion.EulerXYZ(r), _originalRotation);
        transform.localScale = math.float3(_originalScale + s);
    }
}
