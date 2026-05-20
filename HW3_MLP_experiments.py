# ==============================================================================
# HW#3: MLP 신경망 성능에 영향을 주는 요소별 실험과 분석
# 실험 A: 손실 함수 비교   (CrossEntropy vs MSE with softmax)
# 실험 B: 활성화 함수 비교  (ReLU vs LeakyReLU vs Sigmoid)
# 실험 C: 최적화 알고리즘 비교 (SGD / SGD+Momentum / Adam)
# ==============================================================================

# %%
# ==============================================================================
# 0. 라이브러리 임포트 & 공통 설정
# ==============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 재현성 보장 - 이거 없으면 실행할 때마다 결과 달라짐
torch.manual_seed(42)
np.random.seed(42)

# GPU 있으면 CUDA, 없으면 CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 장치: {device}")

# 그래프 기본 스타일
plt.rcParams['figure.dpi'] = 100
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.labelsize'] = 10
sns.set_style("whitegrid")

print("초기 설정 완료")

# %%
# ==============================================================================
# Fashion-MNIST 데이터 로드 (실험 A, C 공용)
# ==============================================================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))   # 픽셀값을 [-1, 1] 범위로 정규화
])

train_fmnist = torchvision.datasets.FashionMNIST('./data', train=True,  download=True, transform=transform)
test_fmnist  = torchvision.datasets.FashionMNIST('./data', train=False, download=True, transform=transform)

# 배치 256 - 코랩 메모리 기준으로 적당한 크기
train_loader_fmnist = DataLoader(train_fmnist, batch_size=256, shuffle=True,  num_workers=2, pin_memory=True)
test_loader_fmnist  = DataLoader(test_fmnist,  batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

print(f"Fashion-MNIST | 학습: {len(train_fmnist):,}개, 테스트: {len(test_fmnist):,}개")


# ==============================================================================
# %%
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 실험 A: 손실 함수 비교 — CrossEntropy vs MSE (with softmax)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 목표: 손실 함수만 바꿔서 학습 속도, 수렴 안정성, 최종 정확도 비교
# 고정 조건: 동일 네트워크, Adam(lr=0.001), 30 에폭, Fashion-MNIST
# ==============================================================================

# %%
# A-1. 모델 정의
class MLP_A(nn.Module):
    """손실 함수 비교 실험용 MLP - 구조 동일하게 고정"""
    def __init__(self):
        super().__init__()
        self.fc1   = nn.Linear(784, 256)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3   = nn.Linear(128, 10)   # 출력층에는 활성화 없음

    def forward(self, x):
        x = x.view(x.size(0), -1)         # 28x28 이미지 → 784차원 벡터로 펼치기
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)

# %%
# A-2. 학습 함수
def train_A(loss_type, epochs=30, lr=0.001):
    """
    loss_type: 'CE' (CrossEntropy) 또는 'MSE'
    - CE는 내부적으로 softmax 포함, logit 그대로 넣으면 됨
    - MSE는 softmax 직접 붙여야 함 + 정답을 one-hot 형태로 변환
    """
    model     = MLP_A().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.CrossEntropyLoss() if loss_type == 'CE' else nn.MSELoss()

    train_losses, test_losses = [], []
    train_accs,  test_accs   = [], []
    grad_norms_fc1 = []   # fc1 레이어 gradient norm - 초반/후반 gradient 흐름 추적

    for ep in range(epochs):
        # ── 학습 ──────────────────────────────────────────────────────────────
        model.train()
        ep_loss, correct, total = 0.0, 0, 0
        batch_grads = []

        for X, y in train_loader_fmnist:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()

            if loss_type == 'CE':
                out  = model(X)
                loss = loss_fn(out, y)
            else:
                # MSE: softmax로 확률값 변환 후 one-hot 정답과 비교
                out  = torch.softmax(model(X), dim=1)
                y_oh = torch.zeros(y.size(0), 10, device=device)
                y_oh.scatter_(1, y.unsqueeze(1), 1.0)   # one-hot 인코딩
                loss = loss_fn(out, y_oh)

            loss.backward()
            batch_grads.append(model.fc1.weight.grad.norm().item())
            optimizer.step()

            ep_loss += loss.item()
            _, pred  = out.max(1)
            total   += y.size(0)
            correct += pred.eq(y).sum().item()

        train_losses.append(ep_loss / len(train_loader_fmnist))
        train_accs.append(100.0 * correct / total)
        grad_norms_fc1.append(np.mean(batch_grads))

        # ── 평가 ──────────────────────────────────────────────────────────────
        model.eval()
        t_loss, t_correct, t_total = 0.0, 0, 0
        with torch.no_grad():
            for X, y in test_loader_fmnist:
                X, y = X.to(device), y.to(device)
                if loss_type == 'CE':
                    out    = model(X)
                    t_loss += loss_fn(out, y).item()
                else:
                    out    = torch.softmax(model(X), dim=1)
                    y_oh   = torch.zeros(y.size(0), 10, device=device)
                    y_oh.scatter_(1, y.unsqueeze(1), 1.0)
                    t_loss += loss_fn(out, y_oh).item()
                _, pred    = out.max(1)
                t_total   += y.size(0)
                t_correct += pred.eq(y).sum().item()

        test_losses.append(t_loss / len(test_loader_fmnist))
        test_accs.append(100.0 * t_correct / t_total)

        if (ep + 1) % 5 == 0:
            print(f"  [{loss_type}] ep {ep+1:3d}/{epochs} | "
                  f"loss: {train_losses[-1]:.4f} | test_acc: {test_accs[-1]:.2f}%")

    return dict(train_loss=train_losses, test_loss=test_losses,
                train_acc=train_accs,   test_acc=test_accs,
                grad_norms=grad_norms_fc1)

# %%
# A-3. 실험 실행
print("=" * 55)
print("실험 A: 손실 함수 비교 시작")
print("=" * 55)
print("\n[CrossEntropy Loss]")
res_CE  = train_A('CE',  epochs=30, lr=0.001)
print("\n[MSE Loss + softmax]")
res_MSE = train_A('MSE', epochs=30, lr=0.001)
print("\n실험 A 완료!")

# %%
# A-4. 시각화
epx = range(1, 31)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Loss 곡선 비교 (실선=Train, 점선=Test)
axes[0].plot(epx, res_CE['train_loss'],  'b-',  lw=2, label='CE  Train')
axes[0].plot(epx, res_CE['test_loss'],   'b--', lw=2, label='CE  Test',  alpha=0.7)
axes[0].plot(epx, res_MSE['train_loss'], 'r-',  lw=2, label='MSE Train')
axes[0].plot(epx, res_MSE['test_loss'],  'r--', lw=2, label='MSE Test',  alpha=0.7)
axes[0].set(xlabel='Epoch', ylabel='Loss', title='Loss 비교')
axes[0].legend(); axes[0].grid(alpha=0.3)

# Test Accuracy 비교
axes[1].plot(epx, res_CE['test_acc'],  'b-', lw=2, label='CrossEntropy')
axes[1].plot(epx, res_MSE['test_acc'], 'r-', lw=2, label='MSE+Softmax')
axes[1].set(xlabel='Epoch', ylabel='Test Accuracy (%)', title='Test Accuracy 비교')
axes[1].legend(); axes[1].grid(alpha=0.3)

# Gradient Norm 비교 — 학습 초반에는 크고 후반에 줄어드는지,
# MSE는 CE보다 gradient가 약해서 학습이 느린지 확인
axes[2].plot(epx, res_CE['grad_norms'],  'b-', lw=2, label='CrossEntropy')
axes[2].plot(epx, res_MSE['grad_norms'], 'r-', lw=2, label='MSE+Softmax')
axes[2].set(xlabel='Epoch', ylabel='FC1 Gradient L2 Norm',
            title='Layer1 Gradient 흐름\n(작으면 gradient vanishing 의심)')
axes[2].legend(); axes[2].grid(alpha=0.3)

plt.suptitle('실험 A: 손실 함수 비교 결과 (Fashion-MNIST)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('exp_A_results.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# A-5. 정량 비교 표
def first_epoch_above(accs, threshold):
    """threshold 정확도를 처음 넘는 에폭 번호"""
    for i, a in enumerate(accs):
        if a >= threshold:
            return i + 1
    return f">{len(accs)}"

df_A = pd.DataFrame([
    {
        '손실 함수':           'CrossEntropy',
        '최종 Test Acc (%)':   f"{res_CE['test_acc'][-1]:.2f}",
        '최솟값 Train Loss':   f"{min(res_CE['train_loss']):.4f}",
        '85% 도달 Epoch':      first_epoch_above(res_CE['test_acc'], 85),
        '최종 Gradient Norm':  f"{res_CE['grad_norms'][-1]:.4f}",
    },
    {
        '손실 함수':           'MSE (with softmax)',
        '최종 Test Acc (%)':   f"{res_MSE['test_acc'][-1]:.2f}",
        '최솟값 Train Loss':   f"{min(res_MSE['train_loss']):.4f}",
        '85% 도달 Epoch':      first_epoch_above(res_MSE['test_acc'], 85),
        '최종 Gradient Norm':  f"{res_MSE['grad_norms'][-1]:.4f}",
    },
])
print("\n[실험 A 정량 비교]")
print(df_A.to_string(index=False))


# ==============================================================================
# %%
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 실험 B: 활성화 함수 비교 — ReLU vs LeakyReLU vs Sigmoid
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 목표: Dead ReLU 발생 유도 + 활성화 함수별 학습 특성 비교
# 고정 조건: 동일 네트워크, Adam(lr=0.01), CrossEntropy, 300 에폭, make_moons
# weight std=0.01로 작게 초기화 → Dead ReLU 발생 유리한 환경 조성
# ==============================================================================

# %%
# B-1. make_moons 데이터 준비
X, y = make_moons(n_samples=2000, noise=0.2, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_tr = scaler.fit_transform(X_tr)   # 학습 데이터 기준으로 정규화
X_te = scaler.transform(X_te)       # 테스트 데이터는 동일한 scaler 적용

# PyTorch 텐서 변환
Xtr_t = torch.FloatTensor(X_tr).to(device)
ytr_t = torch.LongTensor(y_tr).to(device)
Xte_t = torch.FloatTensor(X_te).to(device)
yte_t = torch.LongTensor(y_te).to(device)

train_loader_moons = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=64, shuffle=True)
test_loader_moons  = DataLoader(TensorDataset(Xte_t, yte_t), batch_size=64, shuffle=False)

print(f"make_moons | 학습: {len(X_tr)}개, 테스트: {len(X_te)}개")

# %%
# B-2. 모델 정의 (활성화 함수를 파라미터로 받아서 쉽게 교체)
class MLP_B(nn.Module):
    """
    활성화 함수 비교 실험용 MLP
    - weight std=0.01로 작게 초기화 → Dead ReLU 유도
    - 중간 레이어 출력값 저장 기능 포함 (시각화에 사용)
    """
    def __init__(self, activation='relu'):
        super().__init__()

        def make_act():
            if activation == 'relu':
                return nn.ReLU()
            elif activation == 'leakyrelu':
                return nn.LeakyReLU(negative_slope=0.01)   # 음수 입력에도 0.01배 gradient 통과
            elif activation == 'sigmoid':
                return nn.Sigmoid()

        self.fc1  = nn.Linear(2, 128)
        self.act1 = make_act()
        self.fc2  = nn.Linear(128, 64)
        self.act2 = make_act()
        self.fc3  = nn.Linear(64, 2)

        # weight를 작게 초기화 → 초기 pre-activation 값이 음수로 치우칠 가능성 ↑
        # → ReLU에서 Dead Neuron 발생 가능성 높아짐
        for fc in [self.fc1, self.fc2, self.fc3]:
            nn.init.normal_(fc.weight, mean=0.0, std=0.01)
            nn.init.zeros_(fc.bias)

        # 중간 레이어 출력 저장 (forward 할 때마다 덮어씌워짐)
        self.h1 = None   # act1 통과한 출력
        self.h2 = None   # act2 통과한 출력

    def forward(self, x):
        h1 = self.act1(self.fc1(x))
        h2 = self.act2(self.fc2(h1))
        self.h1 = h1.detach().cpu()   # gradient 계산 필요 없으니 detach
        self.h2 = h2.detach().cpu()
        return self.fc3(h2)

# %%
# B-3. 학습 함수
# 활성화 분포 스냅샷을 3개 시점(초반/중반/후반)에서 저장
SNAP_EPS_B  = [9, 149, 299]          # 10번째, 150번째, 300번째 에폭
SNAP_LABELS = ['Epoch 10', 'Epoch 150', 'Epoch 300']

def train_B(activation, epochs=300, lr=0.01):
    model    = MLP_B(activation).to(device)
    loss_fn  = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, test_losses = [], []
    train_accs,  test_accs   = [], []
    grad_norms_fc1 = []   # Layer 1 gradient
    grad_norms_fc2 = []   # Layer 2 gradient
    activation_snaps = {}  # {epoch: {'h1': ndarray, 'h2': ndarray}}

    for ep in range(epochs):
        # ── 학습 ──────────────────────────────────────────────────────────────
        model.train()
        ep_loss, correct, total = 0.0, 0, 0
        bg1, bg2 = [], []

        for X, y in train_loader_moons:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out  = model(X)
            loss = loss_fn(out, y)
            loss.backward()

            bg1.append(model.fc1.weight.grad.norm().item())
            bg2.append(model.fc2.weight.grad.norm().item())
            optimizer.step()

            ep_loss += loss.item()
            _, pred  = out.max(1)
            total   += y.size(0)
            correct += pred.eq(y).sum().item()

        train_losses.append(ep_loss / len(train_loader_moons))
        train_accs.append(100.0 * correct / total)
        grad_norms_fc1.append(np.mean(bg1))
        grad_norms_fc2.append(np.mean(bg2))

        # ── 평가 + 활성화 스냅샷 수집 ─────────────────────────────────────────
        model.eval()
        t_loss, t_correct, t_total = 0.0, 0, 0
        h1_all, h2_all = [], []

        with torch.no_grad():
            for X, y in test_loader_moons:
                X, y = X.to(device), y.to(device)
                out    = model(X)
                t_loss += loss_fn(out, y).item()
                _, pred    = out.max(1)
                t_total   += y.size(0)
                t_correct += pred.eq(y).sum().item()
                h1_all.append(model.h1.numpy())
                h2_all.append(model.h2.numpy())

        test_losses.append(t_loss / len(test_loader_moons))
        test_accs.append(100.0 * t_correct / t_total)

        if ep in SNAP_EPS_B:
            activation_snaps[ep] = {
                'h1': np.vstack(h1_all),   # (테스트샘플수, 128)
                'h2': np.vstack(h2_all),   # (테스트샘플수, 64)
            }

        if (ep + 1) % 100 == 0:
            print(f"  [{activation:10s}] ep {ep+1:3d}/{epochs} | "
                  f"loss: {train_losses[-1]:.4f} | test_acc: {test_accs[-1]:.2f}%")

    return dict(train_loss=train_losses, test_loss=test_losses,
                train_acc=train_accs,   test_acc=test_accs,
                grad_norms_fc1=grad_norms_fc1, grad_norms_fc2=grad_norms_fc2,
                snaps=activation_snaps, model=model)

# %%
# B-4. 실험 실행
print("=" * 55)
print("실험 B: 활성화 함수 비교 시작")
print("=" * 55)
print("\n[ReLU]")
res_relu   = train_B('relu',      epochs=300, lr=0.01)
print("\n[LeakyReLU]")
res_leaky  = train_B('leakyrelu', epochs=300, lr=0.01)
print("\n[Sigmoid]")
res_sigmoid = train_B('sigmoid',  epochs=300, lr=0.01)
print("\n실험 B 완료!")

B_RESULTS = {
    'ReLU':      res_relu,
    'LeakyReLU': res_leaky,
    'Sigmoid':   res_sigmoid,
}
B_COLORS = {'ReLU': 'steelblue', 'LeakyReLU': 'seagreen', 'Sigmoid': 'tomato'}

# %%
# B-5. 시각화 ① Loss / Accuracy 곡선
epx_B = range(1, 301)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for name, res in B_RESULTS.items():
    axes[0].plot(epx_B, res['train_loss'], color=B_COLORS[name], lw=2, label=name)
    axes[1].plot(epx_B, res['test_acc'],   color=B_COLORS[name], lw=2, label=name)

axes[0].set(xlabel='Epoch', ylabel='Training Loss',  title='Loss 비교')
axes[1].set(xlabel='Epoch', ylabel='Test Accuracy (%)', title='Accuracy 비교')
for ax in axes:
    ax.legend(); ax.grid(alpha=0.3)

plt.suptitle('실험 B: 활성화 함수 비교 — 학습 곡선', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('exp_B_curves.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# B-6. 시각화 ② 결정 경계 (make_moons 2D라서 시각화 가능)
def plot_decision_boundary(model, X, y, title, ax):
    x0_min, x0_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x1_min, x1_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x0_min, x0_max, 300),
                          np.linspace(x1_min, x1_max, 300))
    grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]).to(device)
    model.eval()
    with torch.no_grad():
        prob = torch.softmax(model(grid), dim=1)[:, 1].cpu().numpy()
    prob = prob.reshape(xx.shape)
    ax.contourf(xx, yy, prob, levels=50, cmap='RdYlBu', alpha=0.7)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu',
               edgecolors='black', s=15, linewidths=0.4)
    ax.set_title(title); ax.set_xlabel('X1'); ax.set_ylabel('X2')

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, (name, res) in enumerate(B_RESULTS.items()):
    acc = res['test_acc'][-1]
    plot_decision_boundary(res['model'], X_te, y_te,
                           f"{name}\nTest Acc: {acc:.1f}%", axes[i])
plt.suptitle('실험 B: 결정 경계 비교 (make_moons)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('exp_B_decision_boundary.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# B-7. 시각화 ③ 레이어 출력 분포 변화 (초반 / 중반 / 후반)
# x=0 기준선: ReLU는 여기 기준으로 dead 여부가 갈림
fig, axes = plt.subplots(3, 3, figsize=(15, 12))

for row, (name, res) in enumerate(B_RESULTS.items()):
    for col, (snap_ep, snap_label) in enumerate(zip(SNAP_EPS_B, SNAP_LABELS)):
        ax = axes[row, col]
        h1 = res['snaps'][snap_ep]['h1'].flatten()
        ax.hist(h1, bins=60, color=B_COLORS[name], alpha=0.7, edgecolor='white', linewidth=0.3)
        ax.axvline(0, color='red', linestyle='--', lw=1.5, label='x=0')
        dead_pct = np.mean(h1 <= 0) * 100
        ax.set_title(f"{name} | {snap_label}\nLayer1 출력 분포  (≤0 비율: {dead_pct:.1f}%)")
        ax.set_xlabel('활성화 값')
        ax.set_ylabel('빈도')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

plt.suptitle('실험 B: 학습 단계별 Layer1 활성화 분포 변화\n'
             '(ReLU: x=0 왼쪽 = dead neuron / Sigmoid: 양 끝 포화 = vanishing gradient)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('exp_B_activation_dist.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# B-8. 시각화 ④ Dead Neuron 히트맵 (최종 학습 모델 기준)
# 각 뉴런이 테스트 샘플 중 몇 %에서 0 이하를 출력하는지 시각화
# 빨간색에 가까울수록 "항상 죽어있는" 뉴런

def dead_ratio_per_neuron(snaps, snap_ep):
    """뉴런별 dead (<=0) 비율 반환"""
    h1 = snaps[snap_ep]['h1']   # (N, 128)
    h2 = snaps[snap_ep]['h2']   # (N, 64)
    return (h1 <= 0).mean(axis=0), (h2 <= 0).mean(axis=0)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

for col, (name, res) in enumerate(B_RESULTS.items()):
    d1, d2 = dead_ratio_per_neuron(res['snaps'], SNAP_EPS_B[-1])  # 최종 에폭

    # Layer1: 128개 뉴런 → 8×16 격자로 reshape
    im1 = axes[0, col].imshow(d1.reshape(8, 16), cmap='Reds', vmin=0, vmax=1, aspect='auto')
    axes[0, col].set_title(f"{name}\nLayer1 Dead Ratio  (평균: {d1.mean():.2%})")
    axes[0, col].set_xlabel('뉴런 (열)')
    axes[0, col].set_ylabel('뉴런 (행)')
    plt.colorbar(im1, ax=axes[0, col], fraction=0.046, pad=0.04)

    # Layer2: 64개 뉴런 → 8×8 격자
    im2 = axes[1, col].imshow(d2.reshape(8, 8), cmap='Reds', vmin=0, vmax=1, aspect='auto')
    axes[1, col].set_title(f"{name}\nLayer2 Dead Ratio  (평균: {d2.mean():.2%})")
    axes[1, col].set_xlabel('뉴런 (열)')
    axes[1, col].set_ylabel('뉴런 (행)')
    plt.colorbar(im2, ax=axes[1, col], fraction=0.046, pad=0.04)

plt.suptitle('실험 B: Dead Neuron 히트맵  (붉을수록 항상 0 출력 = dead)\n'
             'Sigmoid는 출력이 (0,1)이라 dead=0, 대신 포화 뉴런이 문제',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('exp_B_dead_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# B-9. 시각화 ⑤ Gradient 흐름 (레이어별 gradient norm 변화)
# gradient가 0에 가까우면 그 레이어는 학습 못 하고 있는 것
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for name, res in B_RESULTS.items():
    axes[0].plot(epx_B, res['grad_norms_fc1'], color=B_COLORS[name], lw=2, label=name)
    axes[1].plot(epx_B, res['grad_norms_fc2'], color=B_COLORS[name], lw=2, label=name)

for ax, layer in zip(axes, ['Layer 1 (fc1)', 'Layer 2 (fc2)']):
    ax.set(xlabel='Epoch', ylabel='Gradient L2 Norm',
           title=f'{layer} Gradient Norm\n(0에 가까우면 vanishing gradient)')
    ax.set_yscale('log')   # 로그 스케일 → vanishing 더 잘 보임
    ax.legend(); ax.grid(alpha=0.3)

plt.suptitle('실험 B: 레이어별 Gradient 흐름  (로그 스케일)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('exp_B_gradient_flow.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# B-10. 정량 비교 표
rows_B = []
for name, res in B_RESULTS.items():
    d1, d2 = dead_ratio_per_neuron(res['snaps'], SNAP_EPS_B[-1])
    rows_B.append({
        '활성화 함수':         name,
        'Layer1 Dead (%)':     f"{d1.mean()*100:.2f}",
        'Layer2 Dead (%)':     f"{d2.mean()*100:.2f}",
        '최종 Test Acc (%)':   f"{res['test_acc'][-1]:.2f}",
        '90% 도달 Epoch':      first_epoch_above(res['test_acc'], 90),
        '최종 FC1 Grad Norm':  f"{res['grad_norms_fc1'][-1]:.5f}",
    })
df_B = pd.DataFrame(rows_B)
print("\n[실험 B 정량 비교]")
print(df_B.to_string(index=False))


# ==============================================================================
# %%
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 실험 C: 최적화 알고리즘 비교 — SGD / SGD+Momentum / Adam
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 목표: 학습률(0.1 / 0.01 / 0.001)과 optimizer 조합 비교
#       + Exponential LR Decay(gamma=0.9) 적용 효과 확인
# 고정 조건: 동일 네트워크, CrossEntropy, Fashion-MNIST, 30 에폭
# ==============================================================================

# %%
# C-1. 모델 정의
class MLP_C(nn.Module):
    """Optimizer 비교 실험용 MLP — 구조 A와 동일"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )
    def forward(self, x):
        return self.net(x.view(x.size(0), -1))

# %%
# C-2. 학습 함수
def train_C(opt_type, lr, use_scheduler=False, epochs=30):
    """
    opt_type: 'SGD' / 'SGD_mom' / 'Adam'
    lr: 학습률 (0.1 / 0.01 / 0.001)
    use_scheduler: ExponentialLR(gamma=0.9) 사용 여부
    """
    model   = MLP_C().to(device)
    loss_fn = nn.CrossEntropyLoss()

    if opt_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif opt_type == 'SGD_mom':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif opt_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)

    # 매 에폭마다 lr *= 0.9 → 학습 후반에 overshooting 줄이는 효과
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9) if use_scheduler else None

    train_losses, test_losses = [], []
    train_accs,  test_accs   = [], []
    lr_log     = [lr]           # 학습률 변화 기록
    grad_norms = []             # gradient 안정성 추적

    for ep in range(epochs):
        # ── 학습 ──────────────────────────────────────────────────────────────
        model.train()
        ep_loss, correct, total = 0.0, 0, 0
        batch_grads = []

        for X, y in train_loader_fmnist:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out  = model(X)
            loss = loss_fn(out, y)
            loss.backward()
            batch_grads.append(model.net[0].weight.grad.norm().item())
            optimizer.step()

            ep_loss += loss.item()
            _, pred  = out.max(1)
            total   += y.size(0)
            correct += pred.eq(y).sum().item()

        if scheduler:
            scheduler.step()
            lr_log.append(optimizer.param_groups[0]['lr'])

        train_losses.append(ep_loss / len(train_loader_fmnist))
        train_accs.append(100.0 * correct / total)
        grad_norms.append(np.mean(batch_grads))

        # ── 평가 ──────────────────────────────────────────────────────────────
        model.eval()
        t_loss, t_correct, t_total = 0.0, 0, 0
        with torch.no_grad():
            for X, y in test_loader_fmnist:
                X, y = X.to(device), y.to(device)
                out    = model(X)
                t_loss += loss_fn(out, y).item()
                _, pred    = out.max(1)
                t_total   += y.size(0)
                t_correct += pred.eq(y).sum().item()

        test_losses.append(t_loss / len(test_loader_fmnist))
        test_accs.append(100.0 * t_correct / t_total)

        if (ep + 1) % 10 == 0:
            cur_lr = optimizer.param_groups[0]['lr']
            sched_tag = "+Sched" if use_scheduler else ""
            print(f"  [{opt_type}{sched_tag} lr={lr}] ep {ep+1:3d}/{epochs} | "
                  f"loss: {train_losses[-1]:.4f} | acc: {test_accs[-1]:.2f}% | cur_lr: {cur_lr:.6f}")

    return dict(train_loss=train_losses, test_loss=test_losses,
                train_acc=train_accs,   test_acc=test_accs,
                lr_log=lr_log, grad_norms=grad_norms)

# %%
# C-3. 실험 실행
LR_LIST  = [0.1, 0.01, 0.001]
OPT_LIST = ['SGD', 'SGD_mom', 'Adam']

print("=" * 55)
print("실험 C: Optimizer 비교 시작 (총 18회 학습)")
print("=" * 55)

res_C      = {}   # 스케줄러 없음
res_C_sched = {}  # 스케줄러 있음

for opt in OPT_LIST:
    for lr in LR_LIST:
        key = f"{opt}_lr{lr}"
        print(f"\n[No Sched] {key}")
        res_C[key]       = train_C(opt, lr, use_scheduler=False, epochs=30)
        print(f"[Sched]    {key}")
        res_C_sched[key] = train_C(opt, lr, use_scheduler=True,  epochs=30)

print("\n실험 C 완료!")

# %%
# C-4. 시각화 ① Optimizer 비교 (lr=0.01 기준)
OPT_COLORS = {'SGD': 'steelblue', 'SGD_mom': 'seagreen', 'Adam': 'tomato'}
OPT_LABELS = {'SGD': 'SGD', 'SGD_mom': 'SGD+Momentum', 'Adam': 'Adam'}
epx_C = range(1, 31)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
lr_ref = 0.01

for opt in OPT_LIST:
    key = f"{opt}_lr{lr_ref}"
    label = OPT_LABELS[opt]
    c     = OPT_COLORS[opt]
    axes[0].plot(epx_C, res_C[key]['train_loss'], color=c, lw=2, label=f"{label}")
    axes[1].plot(epx_C, res_C[key]['test_acc'],   color=c, lw=2, label=f"{label}")

for ax in axes:
    ax.legend(); ax.grid(alpha=0.3)
axes[0].set(xlabel='Epoch', ylabel='Training Loss',     title=f'Loss 비교  (lr={lr_ref}, Scheduler 없음)')
axes[1].set(xlabel='Epoch', ylabel='Test Accuracy (%)', title=f'Accuracy 비교  (lr={lr_ref}, Scheduler 없음)')

plt.suptitle('실험 C: Optimizer 비교 (lr=0.01)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('exp_C_optimizer_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# C-5. 시각화 ② 학습률 영향 — 각 Optimizer마다 lr별 Accuracy 비교
# 너무 크면 overshooting, 너무 작으면 수렴 안 함
LR_COLORS = {0.1: 'tomato', 0.01: 'steelblue', 0.001: 'seagreen'}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, opt in enumerate(OPT_LIST):
    ax = axes[i]
    for lr in LR_LIST:
        key = f"{opt}_lr{lr}"
        ax.plot(epx_C, res_C[key]['test_acc'], color=LR_COLORS[lr], lw=2, label=f"lr={lr}")
    ax.set(xlabel='Epoch', ylabel='Test Accuracy (%)',
           title=f'{OPT_LABELS[opt]}\n학습률별 Test Accuracy')
    ax.legend(); ax.grid(alpha=0.3)

plt.suptitle('실험 C: 학습률이 학습 안정성에 미치는 영향', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('exp_C_lr_effect.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# C-6. 시각화 ③ Exponential LR Decay 효과 (lr=0.01 기준)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for opt in OPT_LIST:
    key = f"{opt}_lr0.01"
    c   = OPT_COLORS[opt]
    label = OPT_LABELS[opt]
    axes[0].plot(epx_C, res_C[key]['test_acc'],       color=c, lw=2, ls='-',  label=f"{label}")
    axes[0].plot(epx_C, res_C_sched[key]['test_acc'], color=c, lw=2, ls='--', alpha=0.8,
                 label=f"{label}+Sched")
    # 학습률 감소 곡선 (scheduler 적용 시)
    lr_vals = res_C_sched[key]['lr_log']
    axes[1].plot(range(len(lr_vals)), lr_vals, color=c, lw=2, label=label)

axes[0].set(xlabel='Epoch', ylabel='Test Accuracy (%)',
            title='Scheduler 유무 비교\n(실선=None, 점선=ExponentialLR)')
axes[0].legend(fontsize=8, ncol=2); axes[0].grid(alpha=0.3)

axes[1].set(xlabel='Epoch', ylabel='Learning Rate',
            title='Exponential LR Decay 진행\n(매 에폭마다 ×0.9)')
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.suptitle('실험 C: Exponential LR Decay (gamma=0.9, lr=0.01 기준)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('exp_C_lr_decay.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# C-7. 시각화 ④ Gradient 흐름 — Optimizer별 gradient 안정성 비교
# gradient가 불규칙하게 튀면 overshooting 징후
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, opt in enumerate(OPT_LIST):
    ax = axes[i]
    for lr in LR_LIST:
        key = f"{opt}_lr{lr}"
        ax.plot(epx_C, res_C[key]['grad_norms'], color=LR_COLORS[lr], lw=2, label=f"lr={lr}")
    ax.set(xlabel='Epoch', ylabel='FC1 Gradient L2 Norm',
           title=f'{OPT_LABELS[opt]}\nGradient Norm (불안정하면 진동 심함)')
    ax.legend(); ax.grid(alpha=0.3)

plt.suptitle('실험 C: Optimizer별 Gradient 흐름 비교', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('exp_C_gradient_flow.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# C-8. 정량 비교 표 ① 학습률별 최종 정확도 전체
print("\n[실험 C 정량 비교 ① — 학습률별 최종 Test Accuracy (Scheduler 없음)]")
rows_C1 = []
for opt in OPT_LIST:
    row = {'Optimizer': OPT_LABELS[opt]}
    for lr in LR_LIST:
        row[f"lr={lr}"] = f"{res_C[f'{opt}_lr{lr}']['test_acc'][-1]:.2f}%"
    rows_C1.append(row)
df_C1 = pd.DataFrame(rows_C1)
print(df_C1.to_string(index=False))

# %%
# C-9. 정량 비교 표 ② Optimizer 상세 비교 (lr=0.01)
print("\n[실험 C 정량 비교 ② — lr=0.01 상세 (Scheduler 없음)]")
rows_C2 = []
for opt in OPT_LIST:
    key = f"{opt}_lr0.01"
    res = res_C[key]
    rows_C2.append({
        'Optimizer':           OPT_LABELS[opt],
        '최종 Test Acc (%)':   f"{res['test_acc'][-1]:.2f}",
        '최솟값 Train Loss':   f"{min(res['train_loss']):.4f}",
        '85% 도달 Epoch':      first_epoch_above(res['test_acc'], 85),
        '최종 Grad Norm':      f"{res['grad_norms'][-1]:.4f}",
    })
df_C2 = pd.DataFrame(rows_C2)
print(df_C2.to_string(index=False))

print("\n[실험 C 정량 비교 ③ — Scheduler 적용 효과 (lr=0.01)]")
rows_C3 = []
for opt in OPT_LIST:
    key = f"{opt}_lr0.01"
    no_s = res_C[key]['test_acc'][-1]
    wi_s = res_C_sched[key]['test_acc'][-1]
    rows_C3.append({
        'Optimizer':             OPT_LABELS[opt],
        'No Scheduler Acc (%)':  f"{no_s:.2f}",
        'With Scheduler Acc (%)': f"{wi_s:.2f}",
        '차이 (Sched-NoSched)':  f"{wi_s - no_s:+.2f}",
    })
df_C3 = pd.DataFrame(rows_C3)
print(df_C3.to_string(index=False))


# ==============================================================================
# %%
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 전체 실험 요약 출력
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ==============================================================================

print("\n" + "=" * 60)
print("전체 실험 결과 요약")
print("=" * 60)
print(f"\n[실험 A]  CrossEntropy: {res_CE['test_acc'][-1]:.2f}%  vs  MSE: {res_MSE['test_acc'][-1]:.2f}%")
print(f"[실험 B]  ReLU: {res_relu['test_acc'][-1]:.2f}%  |  "
      f"LeakyReLU: {res_leaky['test_acc'][-1]:.2f}%  |  "
      f"Sigmoid: {res_sigmoid['test_acc'][-1]:.2f}%")
print(f"[실험 C]  SGD: {res_C['SGD_lr0.01']['test_acc'][-1]:.2f}%  |  "
      f"SGD+Mom: {res_C['SGD_mom_lr0.01']['test_acc'][-1]:.2f}%  |  "
      f"Adam: {res_C['Adam_lr0.01']['test_acc'][-1]:.2f}%  (lr=0.01 기준)")
print("\n모든 그래프가 현재 디렉토리에 png로 저장됨")
print("exp_A_results.png / exp_B_curves.png / exp_B_decision_boundary.png /")
print("exp_B_activation_dist.png / exp_B_dead_heatmap.png / exp_B_gradient_flow.png /")
print("exp_C_optimizer_comparison.png / exp_C_lr_effect.png /")
print("exp_C_lr_decay.png / exp_C_gradient_flow.png")
