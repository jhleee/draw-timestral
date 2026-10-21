# Draw the Future

Draw a line. Let AI guess the rest.

공식 **Mistral Timestral-5M**으로 곡선의 앞 절반만 보고 뒤 절반을 예측하는 브라우저 데모입니다. React, TypeScript, HTML Canvas, ONNX Runtime Web / WebAssembly를 사용합니다.

## 사용

왼쪽 시작 영역에서 오른쪽 끝까지 선을 한 번에 그린 뒤 **Predict the future**를 누릅니다. 처음에는 공식 모델 가중치 약 **20.84MB**와 WebAssembly 런타임을 내려받습니다. 이후 예측은 기기에서 실행하며 별도 서버, 로그인, API 키가 필요 없습니다. 그림 좌표는 업로드하지 않습니다. 첫 다운로드 시간과 추론 속도는 기기와 네트워크에 따라 달라집니다.

가로축 전체를 0–4초로 해석합니다. 실제로 4초 동안 그릴 필요는 없습니다. 128개 등간격 Y 샘플 중 앞 64개만 모델에 전달하고, 뒤 64개는 예측 후 정답 비교에 사용합니다.

- 관측: 청록색 / 실제 정답: 회색 점선 / 모델 중앙값: 주황색
- 불확실성: q10–q90, 중앙 80% 예측 구간
- 지표: MAE, 구간 포함률, q90 초과율
- 선택 비교: 마지막 8개 관측값에 적합한 선형 기준 모델

이 실험은 1차원 곡선 외삽이며 실제 사람의 손동작이나 의도를 예측하는 모델은 아닙니다. 갑작스러운 방향 전환을 맞히지 못할 수 있습니다. 모델 출력을 임의 이동·평활하거나 0–1로 잘라내지 않습니다. 범위를 벗어나면 Y축을 확장합니다. 한 곡선의 포함률로 확률 예측의 보정 성능을 판단하면 안 됩니다.

## 로컬 실행 및 빌드

Node.js 22 이상을 사용합니다.

```bash
npm ci
npm run dev
npm test
npm run build
```

## GitHub Pages

```bash
npm run build:pages
```

이 명령으로 생성한 `docs/`를 소스와 함께 커밋합니다. 저장소 **Settings → Pages → Deploy from a branch → master → /docs**로 배포합니다. `docs/.nojekyll`을 유지하세요. Vite base는 `./`이므로 프로젝트 하위 경로에서도 동작합니다.

## 실제 모델 검증

공식 `TimestralForecaster.forecast`를 고정 입력 `(1,1,64)`과 미래 길이 64로 ONNX에 내보냈습니다. 모델 재학습, 증류, 양자화나 규칙 기반 대체를 사용하지 않습니다. 공식 정규화와 mirrored forecast averaging, 기본 비음수 입력 처리를 유지합니다. 가중치는 원본 safetensors 그대로 다운로드하고 SHA-256을 검사합니다.

- 원본 PyTorch ↔ 네이티브 ONNX: 9종 입력에서 최대 절대 차이 약 `7.2e-7` 이하.
- 실제 ONNX Runtime Web WASM ↔ 원본 PyTorch: 다섯 곡선에서 최대 절대 차이 약 `1.02e-6` 이하.
- 로컬 Node WASM 측정: 약 11–34ms. 모바일 브라우저의 보장 성능이 아닙니다.
- 리샘플링·입력 제약·선형 기준·지표 테스트 5개 통과.

상세 수치는 `browser-model/equivalence.json`, `browser-model/wasm-equivalence.json`에 있습니다. 실제 모바일 기기의 드로잉·렌더링은 별도 검증이 필요합니다. 로딩 또는 추론에 실패하면 오류를 표시하며 가짜 예측을 생성하지 않습니다.

## 모델 출처와 라이선스

- 공식 코드: https://github.com/mistralai/timestral
- 코드 commit: `a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff` (Apache-2.0)
- 모델: https://huggingface.co/mistralai/Timestral-5M
- 가중치 revision: `bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f` (Modified MIT)
- SHA-256: `7ace108c4987e7f2a99a75a445240019d1d31c2117139d1bd05ffdfd25aca738`

모델 가중치는 이 저장소에 포함하지 않습니다. `browser-model/browser-graph.gz.b64`에는 원본 가중치를 외부 데이터로 참조하는 ONNX 그래프를 압축해 포함합니다. 변환 스크립트는 형식 호환을 위한 연산 매핑과 export 수정을 포함하며, 원본과 동등성 검증을 거쳤습니다. 재변환 스크립트 실행에는 Python, 공식 Timestral, PyTorch 2.9.1+cpu, onnx, onnxruntime, numpy가 필요합니다. 앱 사용자는 이 Python 환경이 필요 없습니다.

이 프로젝트의 자체 UI 코드는 MIT 라이선스입니다. 공식 Timestral 코드와 모델의 라이선스는 각 원본 저장소를 따릅니다.
