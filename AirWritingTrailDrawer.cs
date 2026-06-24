using UnityEngine;
using System.Collections;
using System.Collections.Generic;

/// <summary>
/// Air Writing Trail 생성 스크립트
///
/// 실행 방법
/// 1. 기존 AirWritingTrailDrawer.cs를 이 파일로 교체합니다.
/// 2. Manager 오브젝트의 AirWritingTrailDrawer에서
///    - Pen Tip: 검지 끝 PenTip
///    - Trail Prefab: TrailRenderer가 들어있는 TrailObject
///    를 연결합니다.
/// 3. 권장 초기값
///    - Trail Lifetime: 30
///    - Delete Delay: 12
///    - Fade Time: 2.5
///    - Min Vertex Distance: 0.003
///
/// C 키를 누르면 지금까지 생성된 Trail을 모두 지울 수 있습니다.
/// </summary>
public class AirWritingTrailDrawer : MonoBehaviour
{
    [Header("필수 연결")]
    [Tooltip("검지 끝에 배치한 필기 기준 오브젝트")]
    public Transform penTip;

    [Tooltip("TrailRenderer가 들어있는 TrailObject 프리팹")]
    public GameObject trailPrefab;

    [Header("Trail 유지 및 표현")]
    [Tooltip("기존 Inspector 값이 남아 있어도 실행 시 권장값을 적용합니다.")]
    public bool applyRecommendedSettingsOnStart = true;

    [Tooltip("한 획 내부의 점이 유지되는 시간입니다. 너무 작으면 쓰는 도중 앞부분이 사라집니다.")]
    public float trailLifetime = 30f;

    [Tooltip("버튼을 뗀 뒤 Fade를 시작하기 전까지 기다리는 시간입니다.")]
    public float deleteDelay = 12f;

    [Tooltip("선이 서서히 사라지는 시간입니다.")]
    public float fadeTime = 2.5f;

    [Tooltip("점 사이 최소 거리입니다. 작을수록 곡선이 촘촘해집니다.")]
    public float minVertexDistance = 0.003f;

    [Tooltip("완성된 획을 자동 삭제할지 여부입니다.")]
    public bool autoDeleteCompletedTrail = true;

    [Header("사용자 제어")]
    public KeyCode clearAllKey = KeyCode.C;

    [Header("애니메이션")]
    public bool useGripAnimation = false;
    public Animator animator;
    public string gripParameterName = "Grip";

    private TrailRenderer currentTrail;
    private bool isDrawing = false;

    // 생성된 Trail들을 추적해서 C 키로 한 번에 삭제
    private readonly List<GameObject> spawnedTrails = new List<GameObject>();


    private void Awake()
    {
        // 기존 컴포넌트에 저장된 deleteDelay=2.5 같은 값이 남아 있어도
        // 첫 테스트에서는 권장값을 확실하게 적용합니다.
        if (applyRecommendedSettingsOnStart)
        {
            trailLifetime = 30f;
            deleteDelay = 12f;
            fadeTime = 2.5f;
            minVertexDistance = 0.003f;
        }
    }

    void Update()
    {
        if (clearAllKey != KeyCode.None && Input.GetKeyDown(clearAllKey))
        {
            ClearAllTrails();
        }

        if (penTip == null)
        {
            return;
        }

        // 버튼이 눌리는 순간 새 획 시작
        if (UDPReceiver.isDrawing && !isDrawing)
        {
            StartDrawing();
        }

        // 버튼을 누르는 동안 PenTip을 따라감
        if (isDrawing && currentTrail != null)
        {
            currentTrail.transform.position = penTip.position;
        }

        // 버튼을 떼는 순간 획 종료
        if (!UDPReceiver.isDrawing && isDrawing)
        {
            StopDrawing();
        }

        if (useGripAnimation && animator != null)
        {
            animator.SetFloat(
                gripParameterName,
                UDPReceiver.isDrawing ? 1f : 0f
            );
        }
    }

    private void StartDrawing()
    {
        if (trailPrefab == null)
        {
            Debug.LogWarning(
                "Trail Prefab이 비어 있습니다. TrailObject 프리팹을 연결하세요."
            );
            return;
        }

        GameObject newTrail = Instantiate(
            trailPrefab,
            penTip.position,
            Quaternion.identity
        );

        newTrail.SetActive(true);
        newTrail.transform.SetParent(null, true);
        spawnedTrails.Add(newTrail);

        currentTrail = newTrail.GetComponent<TrailRenderer>();

        if (currentTrail == null)
        {
            Debug.LogWarning(
                "Trail Prefab에 TrailRenderer 컴포넌트가 없습니다."
            );
            spawnedTrails.Remove(newTrail);
            Destroy(newTrail);
            return;
        }

        // 프리팹 Inspector 값이 작아도 실행 시 권장값으로 적용
        currentTrail.time = Mathf.Max(0.1f, trailLifetime);
        currentTrail.minVertexDistance = Mathf.Max(
            0.0001f,
            minVertexDistance
        );

        // 이전 프리팹 잔상이나 첫 프레임 직선을 방지
        currentTrail.emitting = false;
        currentTrail.Clear();
        currentTrail.transform.position = penTip.position;

        isDrawing = true;
        StartCoroutine(StartTrailNextFrame(currentTrail));
    }

    private void StopDrawing()
    {
        isDrawing = false;

        TrailRenderer finishedTrail = currentTrail;
        currentTrail = null;

        if (
            finishedTrail != null
            && autoDeleteCompletedTrail
        )
        {
            StartCoroutine(
                DeleteWithDelay(
                    finishedTrail,
                    deleteDelay,
                    fadeTime
                )
            );
        }
    }

    private IEnumerator StartTrailNextFrame(
        TrailRenderer trail
    )
    {
        // 생성된 같은 프레임에서 원점→PenTip 직선이 생기는 것을 방지
        yield return null;

        if (trail != null)
        {
            trail.transform.position = penTip.position;
            trail.Clear();
            trail.emitting = true;
        }
    }

    private IEnumerator DeleteWithDelay(
        TrailRenderer trail,
        float delay,
        float duration
    )
    {
        yield return new WaitForSeconds(
            Mathf.Max(0f, delay)
        );

        if (trail != null)
        {
            yield return FadeAndDestroy(
                trail,
                Mathf.Max(0.01f, duration)
            );
        }
    }

    private IEnumerator FadeAndDestroy(
        TrailRenderer trail,
        float duration
    )
    {
        if (trail == null)
        {
            yield break;
        }

        float originalStartWidth = trail.startWidth;
        float originalEndWidth = trail.endWidth;
        float elapsed = 0f;

        while (elapsed < duration)
        {
            if (trail == null)
            {
                yield break;
            }

            elapsed += Time.deltaTime;
            float ratio = Mathf.Clamp01(
                elapsed / duration
            );

            trail.startWidth = Mathf.Lerp(
                originalStartWidth,
                0f,
                ratio
            );
            trail.endWidth = Mathf.Lerp(
                originalEndWidth,
                0f,
                ratio
            );

            yield return null;
        }

        if (trail != null)
        {
            spawnedTrails.Remove(
                trail.gameObject
            );
            Destroy(trail.gameObject);
        }
    }

    public void ClearAllTrails()
    {
        if (currentTrail != null)
        {
            Destroy(currentTrail.gameObject);
            currentTrail = null;
        }

        for (
            int index = spawnedTrails.Count - 1;
            index >= 0;
            index--
        )
        {
            GameObject trailObject =
                spawnedTrails[index];

            if (trailObject != null)
            {
                Destroy(trailObject);
            }
        }

        spawnedTrails.Clear();
        isDrawing = false;
    }
}
