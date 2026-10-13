# Final Pure BiLSTM model

This directory contains the model pair used by the final glove-to-Unity demonstration.

- `pure_bilstm.pt`: Pure BiLSTM character-recognition weights
- `meta.pkl`: matching label order and normalization metadata

The two files must always be replaced together. The bridge loads them directly from this `bridge/weights/` directory.

## Evaluation

The selected checkpoint scored 2,578/2,600 (99.15%) on the stored balanced evaluation set. This offline result does not guarantee the same accuracy during live glove use; sensor mounting, calibration, writing style, and signal quality affect real-time predictions.

## Provenance

- Model SHA-256: `5CBFB6E8206C4DEE68409AE0A217BD2FA0D03E990ACF88A92FB9391B912F194B`
- Metadata SHA-256: `FD840C4647F1099484269AD8991BFA6572D72BCD666FADA719B0A619367D3C63`
- Selected: 2026-08-24

Training datasets are intentionally not included in this public repository.
