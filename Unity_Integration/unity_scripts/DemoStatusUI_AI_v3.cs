using TMPro;
using UnityEngine;
using UnityEngine.UI;

/// <summary>
/// 시연용 상태 패널 + Writing Area 상단 상태 + 보정 안내 오버레이를 제어합니다.
/// Inspector에서 필요한 UI 오브젝트를 연결하세요.
/// </summary>
public class DemoStatusUI : MonoBehaviour
{
    [Header("왼쪽 상태 패널 텍스트")]
    public TMP_Text connectionText;
    public TMP_Text trackingText;
    public TMP_Text writingText;
    public TMP_Text sensorText;
    public TMP_Text packetText;
    public TMP_Text pointerText;
    public TMP_Text pressureText;

    [Header("AI 인식 결과")]
    [Tooltip("StatusPanel의 RecognitionValue를 연결합니다.")]
    public TMP_Text recognitionText;

    [Tooltip("선택 사항: 전체 인식 문장을 표시할 TMP Text")]
    public TMP_Text recognizedSentenceText;

    [Tooltip("confidence가 실제로 제공될 때만 표시할 TMP Text")]
    public TMP_Text recognitionConfidenceText;

    [Header("왼쪽 상태 점")]
    public Image connectionDot;
    public Image trackingDot;
    public Image writingDot;

    [Header("오른쪽 Writing Area 상태")]
    [Tooltip("현재 Writing Area 오른쪽 위의 LIVE TRACKING 텍스트")]
    public TMP_Text liveTrackingText;

    [Header("보정/오프라인 안내 오버레이")]
    [Tooltip("Writing Area 안의 안내 패널 전체")]
    public GameObject trackingOverlay;

    [Tooltip("예: CALIBRATING SENSOR")]
    public TMP_Text overlayTitle;

    [Tooltip("예: Keep the glove still...")]
    public TMP_Text overlaySubtitle;

    [Header("표시 옵션")]
    public bool showPointerValues = true;
    public bool showPressureRaw = true;
    public bool showOverlayWhenDisconnected = true;

    private readonly Color32 green = new Color32(80, 230, 150, 255);
    private readonly Color32 yellow = new Color32(255, 210, 80, 255);
    private readonly Color32 red = new Color32(255, 95, 95, 255);
    private readonly Color32 gray = new Color32(145, 155, 170, 255);
    private readonly Color32 cyan = new Color32(100, 220, 255, 255);
    private readonly Color32 normalText = new Color32(215, 247, 255, 255);

    private void Update()
    {
        bool connected = UDPReceiver.isConnected;
        bool drawing = UDPReceiver.isDrawing;
        string tracking = UDPReceiver.currentTrackingState ?? "WAITING";
        string source = UDPReceiver.currentPointerSource ?? "UNKNOWN";

        // CONNECTION
        SetText(
            connectionText,
            connected ? "CONNECTED" : "DISCONNECTED",
            connected ? green : red
        );
        SetImageColor(connectionDot, connected ? green : red);

        // TRACKING
        Color32 trackingColor = GetTrackingColor(connected, tracking);
        string formattedTracking = FormatTrackingState(tracking);
        SetText(trackingText, formattedTracking, trackingColor);
        SetImageColor(trackingDot, trackingColor);

        // WRITING
        SetText(
            writingText,
            drawing ? "WRITING ON" : "WRITING OFF",
            drawing ? green : gray
        );
        SetImageColor(writingDot, drawing ? green : gray);

        // SENSOR
        SetText(sensorText, FormatSensorName(source), cyan);

        // PACKET: 정상값은 빨간색이 아니라 밝은 흰색
        if (packetText != null)
        {
            packetText.text = UDPReceiver.currentPacketSize > 0
                ? UDPReceiver.currentPacketSize + " B"
                : "--";
            packetText.color = connected ? normalText : gray;
        }

        // POINTER: 정상값은 청록색
        if (pointerText != null)
        {
            pointerText.gameObject.SetActive(showPointerValues);
            if (showPointerValues)
            {
                pointerText.text =
                    "X " + UDPReceiver.currentPointerX.ToString("+0.00;-0.00;0.00") +
                    "   Y " + UDPReceiver.currentPointerY.ToString("+0.00;-0.00;0.00");
                pointerText.color = connected ? cyan : gray;
            }
        }

        // INPUT RAW
        if (pressureText != null)
        {
            pressureText.gameObject.SetActive(showPressureRaw);
            if (showPressureRaw)
            {
                pressureText.text = UDPReceiver.currentPressureRaw.ToString();
                pressureText.color = connected ? normalText : gray;
            }
        }

        UpdateRecognitionUI();
        UpdateLiveTrackingLabel(connected, tracking);
        UpdateTrackingOverlay(connected, tracking);
    }

    private void UpdateRecognitionUI()
    {
        string latestChar = UDPReceiver.currentRecognizedChar ?? "";
        string sentence = UDPReceiver.currentRecognizedText ?? "";
        float confidence = UDPReceiver.currentRecognitionConfidence;

        string displayValue = latestChar;

        if (string.IsNullOrWhiteSpace(displayValue) && !string.IsNullOrWhiteSpace(sentence))
        {
            displayValue = sentence.Substring(sentence.Length - 1, 1);
        }

        if (recognitionText != null)
        {
            recognitionText.text = string.IsNullOrWhiteSpace(displayValue)
                ? "WAITING"
                : displayValue;
            recognitionText.color = string.IsNullOrWhiteSpace(displayValue)
                ? gray
                : cyan;
        }

        if (recognizedSentenceText != null)
        {
            recognizedSentenceText.text = string.IsNullOrWhiteSpace(sentence)
                ? "No recognized text yet"
                : sentence;
            recognizedSentenceText.color = string.IsNullOrWhiteSpace(sentence)
                ? gray
                : normalText;
        }

        if (recognitionConfidenceText != null)
        {
            bool hasConfidence = confidence >= 0f && confidence <= 1f;
            recognitionConfidenceText.gameObject.SetActive(hasConfidence);

            if (hasConfidence)
            {
                recognitionConfidenceText.text =
                    "CONFIDENCE " + (confidence * 100f).ToString("F1") + "%";
                recognitionConfidenceText.color = normalText;
            }
        }
    }

    private void UpdateLiveTrackingLabel(bool connected, string tracking)
    {
        if (liveTrackingText == null)
        {
            return;
        }

        if (!connected)
        {
            liveTrackingText.text = "● OFFLINE";
            liveTrackingText.color = red;
            return;
        }

        switch (tracking.ToUpperInvariant())
        {
            case "TRACKING":
                liveTrackingText.text = "● LIVE TRACKING";
                liveTrackingText.color = green;
                break;

            case "CALIBRATING":
                liveTrackingText.text = "● CALIBRATING";
                liveTrackingText.color = yellow;
                break;

            case "SETTING_REFERENCE":
                liveTrackingText.text = "● SETTING REFERENCE";
                liveTrackingText.color = yellow;
                break;

            case "SHORT_SENSOR_HOLD":
                liveTrackingText.text = "● SENSOR HOLD";
                liveTrackingText.color = yellow;
                break;

            case "S3_SENSOR_LOST":
            case "S2_SENSOR_LOST":
            case "CONNECTION LOST":
                liveTrackingText.text = "● SENSOR LOST";
                liveTrackingText.color = red;
                break;

            default:
                liveTrackingText.text = "● WAITING";
                liveTrackingText.color = gray;
                break;
        }
    }

    private void UpdateTrackingOverlay(bool connected, string tracking)
    {
        if (trackingOverlay == null)
        {
            return;
        }

        if (!connected)
        {
            trackingOverlay.SetActive(showOverlayWhenDisconnected);
            SetText(overlayTitle, "WAITING FOR SENSOR", red);
            SetText(
                overlaySubtitle,
                "Start the Python bridge and connect the glove.",
                normalText
            );
            return;
        }

        switch (tracking.ToUpperInvariant())
        {
            case "CALIBRATING":
                trackingOverlay.SetActive(true);
                SetText(overlayTitle, "CALIBRATING SENSOR", yellow);
                SetText(
                    overlaySubtitle,
                    "Keep the glove still for a few seconds.",
                    normalText
                );
                break;

            case "SETTING_REFERENCE":
                trackingOverlay.SetActive(true);
                SetText(overlayTitle, "SETTING CENTER REFERENCE", yellow);
                SetText(
                    overlaySubtitle,
                    "Keep the glove facing forward.",
                    normalText
                );
                break;

            case "S3_SENSOR_LOST":
            case "S2_SENSOR_LOST":
            case "CONNECTION LOST":
                trackingOverlay.SetActive(true);
                SetText(overlayTitle, "SENSOR CONNECTION LOST", red);
                SetText(
                    overlaySubtitle,
                    "Check the IMU wiring and Python bridge.",
                    normalText
                );
                break;

            case "TRACKING":
                trackingOverlay.SetActive(false);
                break;

            default:
                trackingOverlay.SetActive(false);
                break;
        }
    }

    private Color32 GetTrackingColor(bool connected, string state)
    {
        if (!connected)
        {
            return red;
        }

        switch (state.ToUpperInvariant())
        {
            case "TRACKING":
                return green;

            case "CALIBRATING":
            case "SETTING_REFERENCE":
            case "SHORT_SENSOR_HOLD":
                return yellow;

            case "S3_SENSOR_LOST":
            case "S2_SENSOR_LOST":
            case "CONNECTION LOST":
                return red;

            default:
                return gray;
        }
    }

    private string FormatTrackingState(string state)
    {
        if (string.IsNullOrEmpty(state))
        {
            return "WAITING";
        }

        return state.Replace("_", " ");
    }

    private string FormatSensorName(string source)
    {
        switch (source)
        {
            case "S2_HAND":
            case "S2_HAND_IMU":
                return "S2 · HAND IMU";

            case "S3_FINGER":
            case "S3_FINGER_IMU":
                return "S3 · FINGER IMU";

            default:
                return string.IsNullOrEmpty(source)
                    ? "--"
                    : source.Replace("_", " ");
        }
    }

    private void SetText(TMP_Text target, string value, Color32 color)
    {
        if (target == null)
        {
            return;
        }

        target.text = value;
        target.color = color;
    }

    private void SetImageColor(Image target, Color32 color)
    {
        if (target != null)
        {
            target.color = color;
        }
    }
}
