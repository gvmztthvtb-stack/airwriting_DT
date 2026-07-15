using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

/// <summary>
/// Python unity_bridge.py가 보내는 UDP JSON을 받아
/// 1) 버튼으로 Trail On/Off 상태를 전달하고
/// 2) 손등 가속도 기반으로 손 모델 회전을 안정화하며
/// 3) 손가락 방향 기반 pointer_x / pointer_y로 손 모델 위치를 이동합니다.
///
/// 이 스크립트는 손 모델의 localPosition을 사용하므로,
/// Right Hand Model이 Main Camera 또는 빈 부모 오브젝트의 자식이어도 동작합니다.
/// </summary>
public class UDPReceiver : MonoBehaviour
{
    // AirWritingTrailDrawer.cs가 이 값을 읽어서 Trail을 켜고 끕니다.
    public static bool isDrawing = false;

    // 나중에 UI에서 연결 상태를 표시할 때 사용할 수 있습니다.
    public static bool isConnected = false;

    // DemoStatusUI.cs에서 읽는 현재 상태값
    public static string currentTrackingState = "WAITING";
    public static string currentPointerSource = "UNKNOWN";
    public static int currentPacketSize = 0;
    public static int currentCalibrated = 0;
    public static int currentButton = 0;
    public static float currentPointerX = 0f;
    public static float currentPointerY = 0f;
    public static int currentPressureRaw = 0;

    // AI 인식 결과
    public static string currentRecognizedText = "";
    public static string currentRecognizedChar = "";
    public static float currentRecognitionConfidence = -1f;
    public static string currentTransport = "DIRECT_SERIAL";

    [Serializable]
    private class BridgePayload
    {
        public string type;
        public int ts;
        public int calibrated;

        // Python에서 -1 ~ +1 범위로 보내는 방향 기반 포인터 좌표
        public float pointer_x;
        public float pointer_y;
        public float pointer_angle_x_deg;
        public float pointer_angle_y_deg;

        // 손 회전 안정화용 raw acceleration
        public float hand_ax;
        public float hand_ay;
        public float hand_az;

        // 참고용 quaternion
        public float hand_qx;
        public float hand_qy;
        public float hand_qz;
        public float hand_qw;
        public float finger_qx;
        public float finger_qy;
        public float finger_qz;
        public float finger_qw;

        public int button;
        public int pressure_raw;
        public int packet_size;
        public string pointer_source;
        public string tracking_state;
        public string writing_input_mode;
        public string transport;

        // main.py WebSocket Relay가 추가로 보내는 AI 결과
        public string recognized_text;
        public string recognized_char;
        public float recognition_confidence;
    }

    [Header("UDP 설정")]
    public int port = 12346;

    [Header("필수 연결")]
    [Tooltip("Unity Hierarchy의 Right Hand Model을 연결합니다.")]
    public Transform hand;

    [Header("기준점 재설정")]
    [Tooltip("첫 데이터 수신 시 손 모델의 회전 기준과 초기 Transform을 설정합니다.")]
    public bool autoCalibrateOnStart = true;

    [Tooltip("Python에서 이미 상대 포인터를 계산하므로 기본 OFF를 권장합니다. ON이면 Unity가 현재 pointer_x/y를 다시 중앙으로 빼므로 실행할 때마다 가용 범위가 달라질 수 있습니다.")]
    public bool useAdditionalUnityPointerCenter = false;

    [Tooltip("실행 중 현재 장갑 위치/자세를 중앙으로 다시 설정하는 키입니다.")]
    public KeyCode recenterKey = KeyCode.R;

    [Header("손 회전 - Raw Acceleration 안정 모드")]
    public bool applyHandRotation = true;

    [Tooltip("높을수록 손 모델 회전이 빠르게 따라옵니다.")]
    public float rotationSmooth = 8f;

    [Tooltip("가속도 방향 자체의 떨림을 줄이는 값입니다.")]
    public float accelDirectionSmooth = 8f;

    [Tooltip("작은 회전은 무시합니다. 단위는 degree입니다.")]
    public float rotationDeadZone = 2f;

    [Tooltip("손 모델의 최대 기울기입니다.")]
    public float maxTiltDegree = 35f;

    [Tooltip("손 모델 기본 방향이 맞지 않을 때 사용하는 고정 보정값입니다.")]
    public Vector3 rotationOffsetEuler = Vector3.zero;

    [Tooltip("회전 방향이 반대인 축은 1을 -1로 바꿉니다.")]
    public Vector3 rotationAxisSign = new Vector3(1f, 1f, 1f);

    [Tooltip("앞뒤 기울기")]
    public bool usePitchX = true;

    [Tooltip("Yaw는 가속도만으로 안정적으로 알 수 없으므로 기본 OFF입니다.")]
    public bool useYawY = false;

    [Tooltip("좌우 기울기")]
    public bool useRollZ = true;

    [Header("손 위치 - 방향 기반 2D 포인터")]
    public bool applyPointerPosition = true;

    [Tooltip("좌우/상하 이동 반응 크기입니다. 1부터 시작하세요.")]
    public Vector2 pointerSensitivity = new Vector2(1f, 1f);

    [Tooltip("좌우 또는 상하가 반대로 움직이면 해당 값을 -1로 바꿉니다.")]
    public Vector2 pointerAxisSign = new Vector2(1f, 1f);

    [Tooltip("작은 포인터 변화는 무시합니다.")]
    public float pointerDeadZone = 0.015f;

    [Tooltip("손이 기준 위치에서 움직일 수 있는 최대 범위입니다. Z는 0으로 두면 깊이는 고정됩니다.")]
    public Vector3 maxHandMoveLocal = new Vector3(0.18f, 0.12f, 0f);

    [Tooltip("높을수록 손 위치가 빠르게 따라옵니다.")]
    public float positionSmooth = 10f;

    [Tooltip("기준 위치에 추가로 더할 오프셋입니다.")]
    public Vector3 handPositionOffset = Vector3.zero;

    [Header("디버그")]
    public bool debugLog = true;
    public float connectionTimeoutSeconds = 1.0f;

    private UdpClient client;
    private Thread receiveThread;
    private volatile bool running;

    private readonly object dataLock = new object();
    private BridgePayload latestData;
    private bool hasData;
    private DateTime lastReceiveUtc;

    private bool calibrated;

    // Unity 씬에서 Play를 누르기 전 배치한 손 모델의 기준 Transform
    private Vector3 baseHandLocalPosition;
    private Quaternion baseHandLocalRotation;

    // R 키 또는 자동 보정 시 저장하는 장갑 기준값
    private Vector2 pointerCenter;
    private Vector3 initialGravity = Vector3.up;
    private Vector3 filteredGravity = Vector3.up;

    private float lastDebugTime;

    private void Start()
    {
        isDrawing = false;
        isConnected = false;
        currentTrackingState = "WAITING";
        currentPointerSource = "UNKNOWN";
        currentPacketSize = 0;
        currentCalibrated = 0;
        currentButton = 0;
        currentPointerX = 0f;
        currentPointerY = 0f;
        currentPressureRaw = 0;
        currentRecognizedText = "";
        currentRecognizedChar = "";
        currentRecognitionConfidence = -1f;
        currentTransport = "DIRECT_SERIAL";
        calibrated = false;
        hasData = false;
        pointerCenter = Vector2.zero;

        if (hand == null)
        {
            Debug.LogError("UDPReceiver: Hand가 비어 있습니다. Right Hand Model을 연결하세요.");
            enabled = false;
            return;
        }

        // 씬에서 미리 배치한 1인칭 손 위치와 회전을 고정 기준으로 저장합니다.
        baseHandLocalPosition = hand.localPosition;
        baseHandLocalRotation = hand.localRotation;

        StartReceiver();
    }

    private void StartReceiver()
    {
        try
        {
            client = new UdpClient(port);
            running = true;

            receiveThread = new Thread(ReceiveLoop);
            receiveThread.IsBackground = true;
            receiveThread.Start();

            Debug.Log("UDPReceiver 시작 / Port: " + port);
        }
        catch (Exception e)
        {
            Debug.LogError("UDPReceiver 시작 실패: " + e.Message);
        }
    }

    private void Update()
    {
        BridgePayload data = null;

        lock (dataLock)
        {
            if (hasData)
            {
                data = latestData;
            }
        }

        // 최근 패킷 수신 시간을 기준으로 연결 상태 계산
        isConnected = hasData &&
                      (DateTime.UtcNow - lastReceiveUtc).TotalSeconds < connectionTimeoutSeconds;

        if (!isConnected)
        {
            currentTrackingState = hasData ? "CONNECTION LOST" : "WAITING";
        }

        if (data == null)
        {
            return;
        }

        // UI에서 표시할 최신 상태값을 메인 스레드에서 갱신합니다.
        currentTrackingState = string.IsNullOrEmpty(data.tracking_state)
            ? (data.calibrated == 1 ? "TRACKING" : "CALIBRATING")
            : data.tracking_state;
        currentPointerSource = string.IsNullOrEmpty(data.pointer_source)
            ? "UNKNOWN"
            : data.pointer_source;
        currentPacketSize = data.packet_size;
        currentCalibrated = data.calibrated;
        currentButton = data.button;
        currentPointerX = data.pointer_x;
        currentPointerY = data.pointer_y;
        currentPressureRaw = data.pressure_raw;

        if (!string.IsNullOrEmpty(data.recognized_text))
        {
            currentRecognizedText = data.recognized_text;
        }

        if (!string.IsNullOrEmpty(data.recognized_char))
        {
            currentRecognizedChar = data.recognized_char;
        }

        currentRecognitionConfidence = data.recognition_confidence;

        if (!string.IsNullOrEmpty(data.transport))
        {
            currentTransport = data.transport;
        }

        // Python 자이로 초기 보정이 끝나기 전에는 손 위치를 움직이지 않습니다.
        if (data.calibrated == 0)
        {
            isDrawing = data.button == 1;
            return;
        }

        isDrawing = data.button == 1;

        if (!calibrated && autoCalibrateOnStart)
        {
            Calibrate(data);
        }

        // Game 창에 포커스가 있는 상태에서 R을 누르면 중앙 재설정
        if (Input.GetKeyDown(recenterKey))
        {
            Calibrate(data);
        }

        if (!calibrated)
        {
            return;
        }

        ApplyStableRotation(data);
        ApplyPointerPosition(data);

        if (debugLog && Time.time - lastDebugTime >= 1f)
        {
            Debug.Log(
                "UDP 연결=" + isConnected +
                " / button=" + data.button +
                " / pointer=(" + data.pointer_x.ToString("F2") + ", " +
                                   data.pointer_y.ToString("F2") + ")" +
                " / packet=" + data.packet_size
            );

            lastDebugTime = Time.time;
        }
    }

    /// <summary>
    /// 현재 장갑 자세를 Unity 손 모델의 회전 기준으로 저장합니다.
    /// pointer_x/y는 Python에서 이미 시작 자세 대비 상대 좌표로 계산되므로,
    /// useAdditionalUnityPointerCenter가 OFF이면 Unity에서는 추가 중앙 보정을 하지 않습니다.
    /// </summary>
    private void Calibrate(BridgePayload data)
    {
        pointerCenter = useAdditionalUnityPointerCenter
            ? new Vector2(data.pointer_x, data.pointer_y)
            : Vector2.zero;

        initialGravity = GetSafeAccelerationDirection(
            data.hand_ax,
            data.hand_ay,
            data.hand_az
        );
        filteredGravity = initialGravity;

        // R을 누르면 손 모델을 씬에서 설정한 중앙 위치로 되돌립니다.
        hand.localPosition = baseHandLocalPosition + handPositionOffset;
        hand.localRotation = baseHandLocalRotation * Quaternion.Euler(rotationOffsetEuler);

        calibrated = true;
        if (useAdditionalUnityPointerCenter)
        {
            Debug.Log("Unity 추가 포인터 중앙 보정 완료: 현재 pointer_x/y를 중심으로 저장했습니다.");
        }
        else
        {
            Debug.Log("Unity 초기화 완료: 포인터 기준은 Python 상대 좌표를 그대로 사용합니다.");
        }
    }

    /// <summary>
    /// 손등 가속도에서 중력 방향을 구해 기준 자세 대비 기울기만 적용합니다.
    /// 자이로 yaw 적분을 직접 사용하지 않기 때문에 제자리 빙글빙글 회전이 줄어듭니다.
    /// </summary>
    private void ApplyStableRotation(BridgePayload data)
    {
        if (!applyHandRotation)
        {
            return;
        }

        Vector3 currentGravity = GetSafeAccelerationDirection(
            data.hand_ax,
            data.hand_ay,
            data.hand_az
        );

        filteredGravity = Vector3.Slerp(
            filteredGravity,
            currentGravity,
            Time.deltaTime * accelDirectionSmooth
        );

        Quaternion relativeRotation = Quaternion.FromToRotation(
            initialGravity,
            filteredGravity
        );

        Vector3 euler = NormalizeEuler(relativeRotation.eulerAngles);

        euler.x = ApplyDeadZone(euler.x, rotationDeadZone);
        euler.y = ApplyDeadZone(euler.y, rotationDeadZone);
        euler.z = ApplyDeadZone(euler.z, rotationDeadZone);

        if (!usePitchX)
        {
            euler.x = 0f;
        }

        if (!useYawY)
        {
            euler.y = 0f;
        }

        if (!useRollZ)
        {
            euler.z = 0f;
        }

        euler = new Vector3(
            euler.x * rotationAxisSign.x,
            euler.y * rotationAxisSign.y,
            euler.z * rotationAxisSign.z
        );

        euler.x = Mathf.Clamp(euler.x, -maxTiltDegree, maxTiltDegree);
        euler.y = Mathf.Clamp(euler.y, -maxTiltDegree, maxTiltDegree);
        euler.z = Mathf.Clamp(euler.z, -maxTiltDegree, maxTiltDegree);

        Quaternion targetRotation =
            baseHandLocalRotation *
            Quaternion.Euler(euler) *
            Quaternion.Euler(rotationOffsetEuler);

        hand.localRotation = Quaternion.Slerp(
            hand.localRotation,
            targetRotation,
            Time.deltaTime * rotationSmooth
        );
    }

    /// <summary>
    /// Python의 pointer_x/y를 손 모델의 local X/Y 위치로 변환합니다.
    /// pointer_x/y는 -1~+1 범위이며, maxHandMoveLocal 범위 밖으로는 나갈 수 없습니다.
    /// Z는 baseHandLocalPosition.z에 고정되어 카메라 뒤로 날아가지 않습니다.
    /// </summary>
    private void ApplyPointerPosition(BridgePayload data)
    {
        if (!applyPointerPosition)
        {
            return;
        }

        // Python이 이미 시작 자세 대비 상대 좌표(-1~+1)를 만들기 때문에
        // 기본값에서는 그 좌표를 그대로 사용합니다.
        // Unity에서 다시 중심값을 빼면 Play를 누를 때의 드리프트 위치가 새 중심이 되어
        // 좌우/상하 가용 범위가 실행마다 달라질 수 있습니다.
        float relativeX = useAdditionalUnityPointerCenter
            ? data.pointer_x - pointerCenter.x
            : data.pointer_x;
        float relativeY = useAdditionalUnityPointerCenter
            ? data.pointer_y - pointerCenter.y
            : data.pointer_y;

        relativeX = ApplyDeadZone(relativeX, pointerDeadZone);
        relativeY = ApplyDeadZone(relativeY, pointerDeadZone);

        relativeX *= pointerSensitivity.x * pointerAxisSign.x;
        relativeY *= pointerSensitivity.y * pointerAxisSign.y;

        // 중앙값과 현재값 차이가 커도 -1~+1 안으로 제한
        relativeX = Mathf.Clamp(relativeX, -1f, 1f);
        relativeY = Mathf.Clamp(relativeY, -1f, 1f);

        Vector3 localMove = new Vector3(
            relativeX * maxHandMoveLocal.x,
            relativeY * maxHandMoveLocal.y,
            0f
        );

        Vector3 targetPosition =
            baseHandLocalPosition +
            handPositionOffset +
            localMove;

        // 축별 최종 범위 한 번 더 제한
        targetPosition.x = Mathf.Clamp(
            targetPosition.x,
            baseHandLocalPosition.x + handPositionOffset.x - maxHandMoveLocal.x,
            baseHandLocalPosition.x + handPositionOffset.x + maxHandMoveLocal.x
        );

        targetPosition.y = Mathf.Clamp(
            targetPosition.y,
            baseHandLocalPosition.y + handPositionOffset.y - maxHandMoveLocal.y,
            baseHandLocalPosition.y + handPositionOffset.y + maxHandMoveLocal.y
        );

        // 깊이값은 항상 씬에 배치한 기준값으로 고정
        targetPosition.z = baseHandLocalPosition.z + handPositionOffset.z;

        hand.localPosition = Vector3.Lerp(
            hand.localPosition,
            targetPosition,
            Time.deltaTime * positionSmooth
        );
    }

    private Vector3 GetSafeAccelerationDirection(float x, float y, float z)
    {
        Vector3 acceleration = new Vector3(x, y, z);

        if (acceleration.sqrMagnitude < 0.000001f)
        {
            return Vector3.up;
        }

        return acceleration.normalized;
    }

    private float ApplyDeadZone(float value, float deadZone)
    {
        return Mathf.Abs(value) < deadZone ? 0f : value;
    }

    private Vector3 NormalizeEuler(Vector3 euler)
    {
        euler.x = NormalizeAngle(euler.x);
        euler.y = NormalizeAngle(euler.y);
        euler.z = NormalizeAngle(euler.z);
        return euler;
    }

    private float NormalizeAngle(float angle)
    {
        while (angle > 180f)
        {
            angle -= 360f;
        }

        while (angle < -180f)
        {
            angle += 360f;
        }

        return angle;
    }

    private void ReceiveLoop()
    {
        IPEndPoint anyIp = new IPEndPoint(IPAddress.Any, port);

        while (running)
        {
            try
            {
                byte[] bytes = client.Receive(ref anyIp);
                string json = Encoding.UTF8.GetString(bytes);
                BridgePayload parsed = JsonUtility.FromJson<BridgePayload>(json);

                lock (dataLock)
                {
                    latestData = parsed;
                    hasData = true;
                    lastReceiveUtc = DateTime.UtcNow;
                }
            }
            catch (ObjectDisposedException)
            {
                break;
            }
            catch (SocketException)
            {
                if (!running)
                {
                    break;
                }
            }
            catch (Exception)
            {
                // Unity API 호출 없이 다음 패킷을 계속 기다립니다.
            }
        }
    }

    private void OnDestroy()
    {
        StopReceiver();
    }

    private void OnApplicationQuit()
    {
        StopReceiver();
    }

    private void StopReceiver()
    {
        if (!running)
        {
            return;
        }

        running = false;
        isConnected = false;
        isDrawing = false;

        try
        {
            client?.Close();
        }
        catch (Exception)
        {
        }

        if (receiveThread != null && receiveThread.IsAlive)
        {
            receiveThread.Join(200);
        }
    }
}
