# Fluid CUDA

Jos Stam의 **Stable Fluids** 방법을 기반으로 만든 2D 유체 시뮬레이션 실험 프로젝트입니다.

CPU 버전과 CUDA GPU 버전을 분리해 두었기 때문에, 두 구현을 각각 따로 컴파일하고 실행하면서 성능을 비교할 수 있습니다.

## 이론적 배경

이 프로젝트는 2D 격자 위에서 연기처럼 보이는 유체의 움직임을 시뮬레이션합니다. 화면에 보이는 연기는 스칼라 값인 **밀도장(density field)** 으로 표현하고, 흐름은 각 격자점의 2차원 **속도장(velocity field)** `(u, v)`로 표현합니다.

기본적으로는 비압축성 Navier-Stokes 방정식의 흐름을 따릅니다.

```text
속도 변화 = 외력 + 확산 + 이류 + 투영
밀도 변화 = 밀도 주입 + 확산 + 이류 + 감쇠
```

시뮬레이션은 주로 다음 단계들로 이루어집니다.

| 단계 | 의미 |
|------|------|
| Source | 마우스 입력 또는 자동 발생기로 밀도/속도를 추가 |
| Diffusion | 밀도나 속도가 주변 격자로 퍼지는 과정 |
| Advection | 현재 속도장을 따라 밀도와 속도가 이동하는 과정 |
| Projection | 속도장의 발산을 줄여 비압축성 유체처럼 보이게 만드는 과정 |
| Dissipation | 연기와 속도가 시간이 지나며 서서히 사라지게 하는 감쇠 |

Stable Fluids의 핵심은 물리적으로 완벽한 정확도보다 **시각적으로 안정적인 시뮬레이션**을 우선한다는 점입니다. 일반적인 전진 방식 업데이트는 시간 간격이 커지면 쉽게 불안정해질 수 있습니다. 반면 이 방법은 이류 단계에서 현재 위치의 값이 어디에서 흘러왔는지를 거꾸로 추적합니다. 이를 **semi-Lagrangian advection**이라고 하며, 비교적 큰 시간 간격에서도 시뮬레이션이 폭발하지 않도록 도와줍니다.

또한 투영 단계에서는 Jacobi 반복법을 사용해 압력장을 근사적으로 구하고, 그 압력장을 이용해 속도장의 발산을 줄입니다. 이 과정 덕분에 유체가 갑자기 부풀거나 줄어드는 것처럼 보이지 않고, 비압축성 유체에 가까운 움직임을 갖게 됩니다.

## CPU 버전과 GPU 버전

CPU 버전은 모든 계산을 일반 C++ 반복문으로 수행합니다. 구조가 비교적 직관적이기 때문에 기준 구현으로 보기 좋습니다.

GPU 버전은 주요 시뮬레이션 단계를 CUDA 커널로 옮긴 버전입니다. 각 격자 셀의 계산을 병렬로 처리할 수 있으므로, 이류, 확산, 투영, 감쇠 같은 연산이 GPU 가속에 적합합니다.

현재 GPU 버전에 적용된 최적화는 다음과 같습니다.

- `diff` 또는 `visc`가 0일 때 불필요한 확산 solve 생략
- 밀도장을 `N * N`개의 OpenGL quad로 그리지 않고 하나의 텍스처로 업로드해 렌더링
- 속도장 표시가 꺼져 있을 때 `u/v` 속도장의 `DeviceToHost` 복사 생략

## 저장소 구조

| 경로 | 설명 |
|------|------|
| `cpu/` | CPU 전용 GLUT 뷰어 |
| `gpu/` | CUDA GPU 전용 GLUT 뷰어 |
| `gpu/solver.cu` | CUDA solver 커널 구현 |
| `gpu/solver.h` | CUDA solver 함수 선언 |
| `gpu/THREE_D_PLAN.md` | GPU 버전을 3D 연기 시뮬레이션으로 확장하기 위한 계획 |

## 요구 사항

Windows 기준:

- NVIDIA CUDA Toolkit
- Visual Studio Build Tools  
  `Desktop development with C++` 워크로드 필요
- freeglut

Anaconda를 사용한다면 freeglut은 다음 명령으로 설치할 수 있습니다.

```powershell
conda install -c conda-forge freeglut
```

현재 빌드 스크립트는 freeglut을 아래 경로에서 찾습니다.

```text
%USERPROFILE%\anaconda3\Library
```

## 빌드 및 실행 방법

저장소를 clone합니다.

```powershell
git clone https://github.com/Do1Yun/Fluid_CUDA.git
cd Fluid_CUDA
```

CPU 버전 빌드 및 실행:

```powershell
cd cpu
powershell -ExecutionPolicy Bypass -File .\build.ps1
.\fluid2d_cpu.exe
```

실행하면 콘솔에서 격자 크기 `N`을 입력받습니다. Enter만 누르면 기본값 `1024`로 실행됩니다.

GPU 버전 빌드 및 실행:

```powershell
cd ..\gpu
powershell -ExecutionPolicy Bypass -File .\build.ps1
.\fluid2d_gpu.exe
```

GPU 버전도 실행 시 같은 방식으로 격자 크기 `N`을 입력받습니다. 여러 해상도를 비교하고 싶다면 프로그램을 다시 실행한 뒤 `256`, `512`, `1024`처럼 원하는 값을 입력하면 됩니다.

## 조작법

| 입력 | 동작 |
|------|------|
| 왼쪽 마우스 드래그 | 속도 추가 |
| 오른쪽 마우스 드래그 | 연기 밀도 추가 |
| `c` | 시뮬레이션 초기화 |
| `p` | 일시정지/재개 |
| `v` | 속도장 표시 켜기/끄기 |
| `q` 또는 `ESC` | 종료 |

## 참고

기본 격자 크기는 각 뷰어의 `SIZE` 값으로 정해져 있지만, 실행할 때 콘솔에서 다른 값을 입력할 수 있습니다.

- `cpu/main.cpp`
- `gpu/main.cu`

`SIZE`가 커질수록 연기는 더 자세하게 표현되지만 계산량과 렌더링 비용도 증가합니다. 이후 확장은 GPU 버전을 중심으로 3D 연기 시뮬레이션까지 진행하는 것을 목표로 합니다.
