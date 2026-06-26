# Relay Troubleshooting Quick Guide

## 연결 실패
- main.py 실행 확인
- WebSocket 주소와 포트 확인
- 기본 주소: `ws://127.0.0.1:12347`

## Unity가 안 움직임
- Relay pointer 로그 확인
- Unity UDP 12346 확인
- Apply Pointer Position 체크
- Use Additional Unity Pointer Center 해제

## 범위 끝에 붙음
- Relay에서 `R`
- 감도 `0.30 → 0.25 → 0.22`

## 방향 반대
- Unity Pointer Axis Sign에서 해당 축만 -1

## Trail이 안 그려짐
- Relay `writing=1` 확인
- PenTip / AirWritingTrailDrawer 연결 확인

## AI 결과 없음
- Relay `[AI 인식]` 로그 확인
- AI 통합 UDPReceiver / DemoStatusUI 적용 확인

## 통합 실패
- main.py와 Relay 종료
- 단독 Bridge + Unity 실행
