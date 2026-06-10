# Stable Fluids 구현 리뷰 및 수정 명세

이 문서는 `cpu/`, `gpu/`, `gpu3d/` 구현을 비교한 결과와 이후 수정 기준을 정리한다. 이후 변경은 아래 명세를 우선 기준으로 삼는다.

## 1. 현재 구현 요약

### CPU 2D

- `cpu/main.cpp`는 Jos Stam 방식의 Stable Fluids를 cell-centered 2D 격자로 구현한다.
- `vel_step -> dens_step -> fade` 순서이며, diffusion/projection의 `lin_solve_cpu`는 배열을 제자리 갱신하는 Gauss-Seidel 성격이다.
- obstacle은 `solid` 마스크로 처리한다. solid 내부 속도는 0으로 만들고, scalar는 주변 fluid 값을 평균한다.
- 렌더링은 OpenGL immediate mode로 모든 cell quad를 CPU에서 그린다.

### GPU 2D

- `gpu/solver.cu`는 CPU solver 흐름을 CUDA 커널로 옮겼다.
- 선형 solve는 ping-pong 버퍼를 쓰는 Jacobi 반복이다. CPU의 in-place 반복과 수치 결과가 완전히 같지는 않다.
- 기존 구조는 매 프레임 source field 전체를 Host -> Device로 복사하고, density/u/v 전체를 Device -> Host로 다시 복사해 CPU에서 색상화했다.
- obstacle이 비어 있어도 non-null `d_solid`를 넘겨 solid branch와 solid 적용 커널을 수행했다.

### GPU 3D

- `gpu3d/solver3d.cu`는 3D scalar density와 `u/v/w` 속도장을 사용한다.
- projection과 diffusion은 3D Jacobi 20회 반복이다.
- viewer는 CUDA ray marching으로 기본 volume을 렌더링한다. 기본 출력에서는 density/u/v/w 전체 Host readback을 하지 않고, raymarch 결과 RGBA buffer만 OpenGL texture로 업로드한다.
- 기본 데모는 6면체 박스 안에서 상단 중앙의 고정 원형 smoke inlet이 아래로 바람을 넣고, 이동 가능한 구형 물체에 부딪힌 연기가 퍼지는 풍동 형태이다.

## 2. 어색하거나 올바르지 못한 지점

### 수치적 일관성

- CPU와 GPU의 `lin_solve` 방식이 다르다. CPU는 사실상 Gauss-Seidel, GPU는 Jacobi이다. 같은 반복 횟수 20회를 쓰면 GPU가 더 덜 수렴한다.
- `N=1024` 2D 또는 `N=64~128` 3D에서 projection 20회는 divergence를 충분히 낮추기 어렵다. 시각적으로 압축성처럼 보이는 흐름, 벽 주변 어색한 압력, smoke 뭉침이 생길 수 있다.
- 현재 속도는 cell-centered이다. Stable Fluids 데모로는 충분하지만, 벽/장애물의 no-slip 또는 no-through 조건을 물리적으로 더 정확하게 처리하려면 MAC staggered grid 또는 face fraction 방식이 필요하다.

### Boundary와 obstacle

- solid scalar를 0으로 고정하지 않고 주변 평균으로 채우는 방식은 샘플링 안정성에는 도움이 되지만, smoke가 벽 내부에 숨어 있다가 obstacle 삭제 후 다시 나타나는 느낌을 줄 수 있다.
- solid 근처 projection은 이웃 solid velocity를 0으로 간주한다. 간단한 마스크 처리로는 충분하지만 복잡한 obstacle 경계에서는 압력 경계가 거칠다.
- 2D 자동 smoke source는 현재 `auto_smoke_velocity = -0.75f`라 화면 좌표 기준으로 완만한 아래쪽 흐름을 만든다. 3D 자동 source는 상단 중앙에 고정된 원형 inlet에서 아래 방향으로 흐른다.

### Advection 품질

- semi-Lagrangian advection은 안정적이지만 강하게 dissipative하다. 선명한 smoke나 소용돌이를 원하면 MacCormack/BFECC 보정 또는 vorticity confinement이 필요하다.
- 현재는 CFL 기반 substep 또는 max velocity clamp가 없다. Stable Fluids 특성상 폭발은 잘 안 나지만, 큰 brush impulse에서는 시각적으로 과한 backtrace와 뭉개짐이 생길 수 있다.

### GPU 성능 구조

- 전체 field Host/Device 왕복은 GPU solver 성능을 크게 갉아먹는다. 특히 3D는 `density + u + v + w` 전체 readback 때문에 렌더링 비용이 solver보다 커질 수 있다.
- Jacobi 반복마다 boundary 커널이 여러 번 launch된다. launch overhead와 global memory bandwidth가 병목이다.
- obstacle이 없을 때도 solid 처리를 수행하면 기본 데모 성능이 불필요하게 떨어진다.
- 3D viewer는 GPU-first 계획과 달리 아직 CPU 렌더링 경로에 크게 묶여 있다.

## 3. 이번 수정에서 반영한 사항

### GPU 2D

- source field 작성 경로를 GPU 커널로 이동했다.
  - 매 프레임 `h_dens_prev/h_u_prev/h_v_prev` 전체를 CPU에서 0으로 채우고 복사하던 흐름을 제거했다.
  - 마우스 brush와 자동 smoke source는 작은 CUDA 커널로 `d_*_prev`에 직접 주입한다.
- density 색상화를 GPU 커널로 이동했다.
  - `d_dens/d_u/d_v`를 전부 Host로 가져오지 않고, GPU에서 RGBA pixel buffer를 만든 뒤 `N*N*4`만 복사한다.
  - 속도 벡터 표시가 켜진 경우에만 `u/v`를 Host로 복사한다.
- obstacle이 없으면 solver에 `NULL` solid pointer를 넘겨 solid branch와 solid 적용 커널을 생략한다.
- obstacle cell count를 추적해, 지우개 brush로 모든 obstacle을 지워도 solid 경로가 자동으로 비활성화된다.
- obstacle 변경 시 전체 field를 Host에서 Device로 덮어쓰지 않고, device에서 solid cell만 0으로 정리한다.
- `vel_step`의 `u/v` source add를 하나의 fused kernel로 합쳤다.
- 2D boundary corner 전용 launch를 제거하고 side boundary 커널에서 corner를 직접 계산한다.
- hot kernel 포인터에 `__restrict__`를 추가했다.
- 2D 기본 자동 연기 source의 vertical velocity를 `-2.0f`에서 `-0.75f`로 낮춰 초기 낙하 속도를 줄였다.
- `gpu/build.ps1`에 `--use_fast_math`를 추가했다.

### GPU 3D

- `u/v/w` source add를 하나의 fused kernel로 합쳤다.
- 3D 기본 자동 source를 켜고, 박스 상단 중앙에 고정된 원형 smoke inlet에서 아래로 흐르도록 변경했다. 마우스 휠은 source 위치를 바꾸지 않는다.
- 3D viewer의 도메인 반경을 `domain_half_extent = 1.45f`로 키워 박스 전체 크기를 더 확대했다.
- smoke source 양, inlet 반경, raymarch alpha를 키워 기본 연기량과 가시성을 높였다.
- 이동 가능한 구형 장애물을 추가했다. `m` 또는 `tab` 키로 카메라 이동 모드와 물체 이동 모드를 전환하고, 물체 이동 모드에서는 `WASD/QE`로 obstacle 위치를 바꾼다.
- solver는 현재 obstacle 내부의 density/u/v/w를 0으로 만들고, 장애물 주변 shell에서는 표면 안쪽으로 파고드는 속도 성분을 제거해 연기가 물체를 타고 흐르게 한다.
- 카메라를 따라오는 z slice, velocity vector, floor grid, 입력 평면 guide는 기본 출력에서 제거했다. 기본 화면에는 smoke volume, 6면체 박스, obstacle, wind inlet marker만 남긴다.
- 현재 기본 volume은 CUDA ray marching으로 카메라 ray와 6면체 박스의 교차 구간을 적분해 렌더링한다. 화면정렬 billboard/slice-stack 방식은 사용하지 않는다.
- obstacle을 먼저 그린 뒤, raymarch 커널이 obstacle 전면 hit 지점까지만 연기를 적분해 합성한다. 따라서 연기가 물체보다 카메라에 가까운 경우에는 연기가 물체를 가리고, 물체 뒤쪽 연기는 물체 위에 잘못 덮이지 않는다.
- CUDA raymarch 카메라와 OpenGL 박스/장애물 카메라는 같은 FOV, near/far, forward/right/up frame을 사용해야 한다. OpenGL `gluLookAt`도 world-up 고정값이 아니라 `camera_basis()`의 up vector를 사용한다. 매 frame viewport/projection/render state를 명시적으로 재설정한다.
- `gpu3d/build.ps1`에 `--use_fast_math`를 추가했다.

## 4. 이후 수정 명세

### 수치 검증

- headless regression test를 추가한다.
  - 2D: 작은 `N=16/32`에서 CPU와 GPU 결과 차이를 비교한다.
  - 3D: density non-negativity, projection 전후 divergence 감소량, boundary ghost cell 값을 검사한다.
- projection 후 divergence L2/max norm을 측정하는 CUDA reduction을 추가한다. CUB `DeviceReduce` 사용을 우선 검토한다.
- pressure iteration 수는 고정 20이 아니라 grid 크기와 목표 residual에 따라 조절할 수 있게 만든다.
- CPU/GPU 비교가 목표인 테스트에서는 CPU도 Jacobi 기준 구현을 별도로 두거나, GPU에 red-black Gauss-Seidel을 도입해 반복 의미를 맞춘다.

### Solver 개선

- 기본 Jacobi는 유지하되, 다음 순서로 개선한다.
  1. red-black Gauss-Seidel 또는 weighted Jacobi
  2. residual 기반 early stop
  3. multigrid pressure solve
- periodic boundary가 허용되는 데모 모드라면 cuFFT 기반 Poisson solve를 검토한다.
- 고정 obstacle/벽 조건을 더 정확히 다룰 필요가 있으면 cuSPARSE/cuSOLVER로 sparse Poisson solve를 구성한다.
- smoke detail이 목표라면 vorticity confinement과 MacCormack/BFECC advection을 추가한다.

### GPU 렌더링/전송

- 2D는 다음 단계에서 CUDA/OpenGL PBO interop로 `d_density_pixels -> h_density_pixels -> glTexSubImage2D` 복사를 제거한다.
- 3D는 다음 단계에서 CUDA/OpenGL PBO interop 또는 OpenGL 3D texture 기반 렌더링으로 전환한다.
  - 현재 CUDA ray marching 결과는 2D RGBA buffer로 만든 뒤 Host를 거쳐 OpenGL texture로 업로드한다.
  - 다음 단계에서는 CUDA output buffer를 PBO로 직접 연결해 `d_volume_pixels -> h_volume_pixels -> glTexSubImage2D` 복사를 제거한다.
  - density/u/v/w 전체 Host readback은 기본 출력에서 금지하고, 별도 디버그/검증 모드에서만 허용한다.
- advection sampling은 CUDA texture object를 검토한다. hardware interpolation을 쓰면 bilinear/trilinear 샘플 코드와 global load 수를 줄일 수 있다.
- 필요한 Host 전송이 남는 경우 pinned host memory와 async copy/stream을 사용한다.

### 성능 측정

- solver 시간과 render/upload 시간을 분리해 출력한다.
- Nsight Systems로 kernel launch 수, memcpy, CPU/GPU sync 지점을 먼저 확인한다.
- Nsight Compute로 Jacobi/project/advection의 memory throughput과 occupancy를 확인한다.
- 벡터 표시, obstacle 유무, auto source 유무를 분리한 benchmark를 만든다.

## 5. 검증 상태

- `gpu/build.ps1`: 빌드 성공.
- `gpu3d/build.ps1`: 빌드 성공.
- GUI 실행 검증은 하지 않았다. 두 실행 파일 모두 interactive GLUT viewer라 자동 회귀 테스트가 필요하다.
