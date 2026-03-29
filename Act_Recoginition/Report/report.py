import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import itertools

# ── Suppress warnings ───────────────────────────────────────────────────
import warnings
warnings.filterwarnings('ignore')

print("Building visual report...")

# ── ResNet ──────────────────────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=7):
        super().__init__()
        pad = kernel // 2
        self.conv1    = nn.Conv1d(in_ch, out_ch, kernel, padding=pad)
        self.bn1      = nn.BatchNorm1d(out_ch)
        self.conv2    = nn.Conv1d(out_ch, out_ch, kernel, padding=pad)
        self.bn2      = nn.BatchNorm1d(out_ch)
        self.relu     = nn.ReLU()
        self.shortcut = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))

class ResNet1D(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            ResBlock(in_channels, 64),
            ResBlock(64, 128),
            ResBlock(128, 128),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Linear(128, num_classes)
    def forward(self, x):
        return self.classifier(self.network(x).squeeze(-1))
    def get_features(self, x):
        return self.network(x).squeeze(-1)

# ── Load data ───────────────────────────────────────────────────────────
X_pamap   = np.load('data/X.npy')
y_pamap   = np.load('data/y.npy')
s_pamap   = np.load('data/subjects.npy')
X_phys    = np.load('data/physionet_X.npy')
y_phys    = np.load('data/physionet_y.npy')
s_phys    = np.load('data/physionet_pids.npy')

PAMAP2_ACTIVITIES  = {1:'lying',2:'sitting',3:'standing',4:'walking',
                      5:'running',6:'cycling',7:'nordic_walking'}
PHYSIO_ACTIVITIES  = {0:'sitting',1:'stair_climbing',2:'walking_6mwt',
                      3:'treadmill_walking',4:'timed_up_and_go',5:'cycling'}
SHARED_ACTIVITIES  = {0:'sitting',1:'walking',2:'running',3:'cycling',
                      4:'stair_climbing',5:'treadmill_walking',
                      6:'timed_up_and_go',7:'nordic_walking'}
PAMAP2_TO_SHARED   = {2:0,4:1,5:2,6:3,7:7,1:0,3:0}
PHYSIO_TO_SHARED   = {0:0,2:1,3:5,5:3,1:4,4:6}

# ── Load models ─────────────────────────────────────────────────────────
le = LabelEncoder(); le.fit(y_pamap)
pamap_model = ResNet1D(3, 7)
pamap_model.load_state_dict(torch.load('data/model.pth', weights_only=True))
pamap_model.eval()

harnet = torch.hub.load('OxWearables/ssl-wearables','harnet10',
                        class_num=6, pretrained=False)
harnet.load_state_dict(torch.load('data/harnet_physionet.pth', weights_only=True))
harnet.eval()

# ── Rebuild test predictions ─────────────────────────────────────────────
y_pamap_shared = np.array([PAMAP2_TO_SHARED[l] for l in y_pamap])
y_phys_shared  = np.array([PHYSIO_TO_SHARED[l]  for l in y_phys])
X_all = np.concatenate([X_pamap, X_phys])
y_all = np.concatenate([y_pamap_shared, y_phys_shared])
s_all = np.concatenate([np.array([f'pamap_{s}' for s in s_pamap]),
                         np.array([f'phys_{s}'  for s in s_phys])])

all_subjects = np.unique(s_all)
np.random.seed(42); np.random.shuffle(all_subjects)
n_test    = max(1, int(len(all_subjects)*0.2))
test_subs = set(all_subjects[-n_test:])
test_mask = np.array([s in test_subs for s in s_all])
train_mask= ~test_mask

X_train,y_train = X_all[train_mask],y_all[train_mask]
X_test, y_test  = X_all[test_mask], y_all[test_mask]

def extract_features(X):
    t = torch.FloatTensor(X).permute(0,2,1)
    with torch.no_grad():
        fr = pamap_model.get_features(t).numpy()
        fh = harnet.feature_extractor(t).mean(dim=-1).numpy()
    return fr, fh

fr_tr,fh_tr = extract_features(X_train)
fr_te,fh_te = extract_features(X_test)
sr,sh = StandardScaler(),StandardScaler()
fr_tr=sr.fit_transform(fr_tr); fr_te=sr.transform(fr_te)
fh_tr=sh.fit_transform(fh_tr); fh_te=sh.transform(fh_te)
Xtr=np.concatenate([fr_tr,fh_tr],axis=1)
Xte=np.concatenate([fr_te,fh_te],axis=1)

num_classes  = len(SHARED_ACTIVITIES)
class_counts = np.bincount(y_train, minlength=num_classes)
class_counts = np.where(class_counts==0,1,class_counts)
class_weights= torch.FloatTensor(1.0/class_counts)
class_weights= class_weights/class_weights.sum()*num_classes

fusion_clf = nn.Sequential(
    nn.Linear(Xtr.shape[1],512),nn.BatchNorm1d(512),nn.ReLU(),nn.Dropout(0.4),
    nn.Linear(512,256),nn.BatchNorm1d(256),nn.ReLU(),nn.Dropout(0.3),
    nn.Linear(256,128),nn.ReLU(),nn.Linear(128,num_classes)
)
fusion_clf.load_state_dict(torch.load('data/fusion_model_proper.pth', weights_only=True))
fusion_clf.eval()

with torch.no_grad():
    final_preds = fusion_clf(torch.FloatTensor(Xte)).argmax(1).numpy()

test_classes = np.unique(y_test)
test_labels  = [SHARED_ACTIVITIES[i] for i in test_classes]
f1_fusion    = f1_score(y_test,final_preds,average='macro',labels=test_classes)
acc_fusion   = (final_preds==y_test).mean()

# PAMAP2 model preds
le2 = LabelEncoder(); le2.fit(y_pamap)
from sklearn.model_selection import train_test_split
_,Xtp,_,ytp = train_test_split(X_pamap,le2.transform(y_pamap),
                                test_size=0.2,random_state=42,
                                stratify=le2.transform(y_pamap))
with torch.no_grad():
    pp = pamap_model(torch.FloatTensor(Xtp).permute(0,2,1)).argmax(1).numpy()
f1_pamap  = f1_score(ytp,pp,average='macro')
acc_pamap = (pp==ytp).mean()

# HARNet preds
_,Xph,_,yph = train_test_split(X_phys,y_phys,
                                test_size=0.2,random_state=42,stratify=y_phys)
with torch.no_grad():
    ph = harnet(torch.FloatTensor(Xph).permute(0,2,1)).argmax(1).numpy()
f1_harnet  = f1_score(yph,ph,average='macro')
acc_harnet = (ph==yph).mean()

# ════════════════════════════════════════════════════════
# FIGURE 1 — Pipeline Overview
# ════════════════════════════════════════════════════════
fig1, ax = plt.subplots(figsize=(20, 10))
ax.set_xlim(0, 20); ax.set_ylim(0, 10)
ax.axis('off')
fig1.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')

def box(ax, x, y, w, h, color, text, fontsize=10, textcolor='white'):
    rect = FancyBboxPatch((x-w/2, y-h/2), w, h,
                          boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor='white',
                          linewidth=1.5, zorder=3)
    ax.add_patch(rect)
    ax.text(x, y, text, ha='center', va='center',
            color=textcolor, fontsize=fontsize,
            fontweight='bold', zorder=4,
            wrap=True, multialignment='center')

def arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle='->', color='#58a6ff',
                               lw=2.5), zorder=5)

# Data sources
box(ax,2,8,3,1.2,'#1f6feb','UK Biobank\n700K person-days',11)
box(ax,2,6,3,1.2,'#388bfd','PAMAP2\n9 subjects · 1,347 windows',10)
box(ax,2,4,3,1.2,'#58a6ff','PhysioNet\n39 patients · 5,412 windows',10)

# Models
box(ax,7,6,3,1.2,'#2ea043','ResNet-1D\n(PAMAP2 trained)\nAcc=88.9%',10)
box(ax,7,4,3,1.2,'#3fb950','HARNet10 SSL\n(Pre-trained+Fine-tuned)\nAcc=73.3%',10)

# SSL pretraining
box(ax,7,8,3,1.2,'#6e40c9','Self-Supervised\nPre-training\n(SSL)',10)

# Feature extraction
box(ax,12,5,3,1.2,'#d29922','Feature\nExtraction\n128+1024 dims',10)

# Normalization
box(ax,15.5,5,2.5,1.2,'#f78166','StandardScaler\nNormalization',10)

# Fusion
box(ax,18.5,5,2.5,1.5,'#bc8cff','Fusion\nClassifier\n8 Activities',11)

# Arrows
arrow(ax,2,8,5.5,8); arrow(ax,5.5,8,5.5,6.6); arrow(ax,3.5,6,5.5,6)
arrow(ax,3.5,4,5.5,4); arrow(ax,5.5,4,5.5,4.6)
arrow(ax,8.5,6,10.5,5.3); arrow(ax,8.5,4,10.5,4.7)
arrow(ax,13.5,5,14.25,5); arrow(ax,16.75,5,17.25,5)

# Labels
for txt,x,y,c in [
    ('700K person-days\nPre-training',4.5,8.5,'#8b949e'),
    ('Transfer\nLearning',5,7.2,'#8b949e'),
    ('Fine-tuning\non PhysioNet',5,3.3,'#8b949e'),
    ('1152-dim\nfused vector',14,5.6,'#8b949e'),
]:
    ax.text(x,y,txt,ha='center',va='center',color=c,fontsize=8,style='italic')

ax.text(10,9.5,'Human Activity Recognition — Full Pipeline',
        ha='center',va='center',color='white',fontsize=16,fontweight='bold')
ax.text(10,9.0,f'Total: 6,759 windows · 47 subjects · 8 activities',
        ha='center',va='center',color='#8b949e',fontsize=12)

plt.tight_layout()
plt.savefig('data/fig1_pipeline.png',dpi=150,bbox_inches='tight',
            facecolor='#0d1117')
print("✅ Fig 1 saved")

# ════════════════════════════════════════════════════════
# FIGURE 2 — Dataset Statistics
# ════════════════════════════════════════════════════════
fig2, axes = plt.subplots(1, 3, figsize=(18, 6))
fig2.patch.set_facecolor('#0d1117')
colors_p = ['#388bfd','#58a6ff','#79c0ff','#a5d6ff',
            '#cae8ff','#1f6feb','#0d419d']
colors_ph= ['#3fb950','#2ea043','#56d364','#7ee787',
            '#26a641','#1a7f37','#116329']

for ax in axes:
    ax.set_facecolor('#161b22')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('#30363d')
    ax.spines['left'].set_color('#30363d')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.label.set_color('white')
    ax.xaxis.label.set_color('white')
    ax.title.set_color('white')

# PAMAP2 activity distribution
pamap_counts = [np.sum(y_pamap==i) for i in PAMAP2_ACTIVITIES.keys()]
bars = axes[0].bar(range(len(PAMAP2_ACTIVITIES)),pamap_counts,
                   color=colors_p,edgecolor='#0d1117',linewidth=0.5)
axes[0].set_xticks(range(len(PAMAP2_ACTIVITIES)))
axes[0].set_xticklabels(list(PAMAP2_ACTIVITIES.values()),
                         rotation=40,ha='right',fontsize=9,color='white')
axes[0].set_title('PAMAP2 Dataset\n(Healthy Adults, Lab)',
                  fontsize=12,fontweight='bold')
axes[0].set_ylabel('Windows',color='white')
for bar,val in zip(bars,pamap_counts):
    axes[0].text(bar.get_x()+bar.get_width()/2,bar.get_height()+2,
                 str(val),ha='center',va='bottom',color='white',fontsize=9)

# PhysioNet activity distribution
phys_counts = [np.sum(y_phys==i) for i in PHYSIO_ACTIVITIES.keys()]
bars2 = axes[1].bar(range(len(PHYSIO_ACTIVITIES)),phys_counts,
                    color=colors_ph,edgecolor='#0d1117',linewidth=0.5)
axes[1].set_xticks(range(len(PHYSIO_ACTIVITIES)))
axes[1].set_xticklabels(list(PHYSIO_ACTIVITIES.values()),
                         rotation=40,ha='right',fontsize=9,color='white')
axes[1].set_title('PhysioNet Dataset\n(Elderly Patients, Clinical)',
                  fontsize=12,fontweight='bold')
axes[1].set_ylabel('Windows',color='white')
for bar,val in zip(bars2,phys_counts):
    axes[1].text(bar.get_x()+bar.get_width()/2,bar.get_height()+2,
                 str(val),ha='center',va='bottom',color='white',fontsize=9)

# Combined pie chart
sizes  = [len(X_pamap), len(X_phys)]
clrs   = ['#388bfd','#3fb950']
explode= (0.05, 0.05)
wedges,texts,autotexts = axes[2].pie(
    sizes, explode=explode, colors=clrs,
    autopct='%1.1f%%', startangle=90,
    textprops={'color':'white','fontsize':12},
    wedgeprops={'edgecolor':'#0d1117','linewidth':2}
)
for at in autotexts:
    at.set_fontsize(13)
    at.set_fontweight('bold')
axes[2].set_title(f'Combined Dataset\n{sum(sizes):,} total windows',
                  fontsize=12,fontweight='bold')
axes[2].legend(['PAMAP2 (1,347)','PhysioNet (5,412)'],
               loc='lower center',fontsize=10,
               facecolor='#161b22',labelcolor='white',
               framealpha=0.8,ncol=2)

fig2.suptitle('Dataset Overview',fontsize=16,
              fontweight='bold',color='white',y=1.02)
plt.tight_layout()
plt.savefig('data/fig2_datasets.png',dpi=150,bbox_inches='tight',
            facecolor='#0d1117')
print("✅ Fig 2 saved")

# ════════════════════════════════════════════════════════
# FIGURE 3 — Model Comparison Bar Chart
# ════════════════════════════════════════════════════════
fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))
fig3.patch.set_facecolor('#0d1117')

models  = ['ResNet\n(PAMAP2)', 'HARNet10\n(PhysioNet)', 'Fusion Model\n(All Data)']
accs    = [acc_pamap*100, acc_harnet*100, acc_fusion*100]
f1s     = [f1_pamap, f1_harnet, f1_fusion]
colors3 = ['#388bfd','#3fb950','#bc8cff']

for ax in axes3:
    ax.set_facecolor('#161b22')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('#30363d')
    ax.spines['left'].set_color('#30363d')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')

bars3a = axes3[0].bar(models,accs,color=colors3,
                       edgecolor='#0d1117',linewidth=0.5,width=0.5)
axes3[0].set_ylim(0,110)
axes3[0].set_ylabel('Accuracy (%)',color='white')
axes3[0].set_title('Accuracy Comparison',fontsize=13,fontweight='bold')
axes3[0].tick_params(axis='x',colors='white',labelsize=10)
axes3[0].tick_params(axis='y',colors='white')
for bar,val in zip(bars3a,accs):
    axes3[0].text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.5,
                  f'{val:.1f}%',ha='center',va='bottom',
                  color='white',fontsize=12,fontweight='bold')

bars3b = axes3[1].bar(models,f1s,color=colors3,
                       edgecolor='#0d1117',linewidth=0.5,width=0.5)
axes3[1].set_ylim(0,1.15)
axes3[1].set_ylabel('Macro F1 Score',color='white')
axes3[1].set_title('F1 Score Comparison',fontsize=13,fontweight='bold')
axes3[1].tick_params(axis='x',colors='white',labelsize=10)
axes3[1].tick_params(axis='y',colors='white')
for bar,val in zip(bars3b,f1s):
    axes3[1].text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.01,
                  f'{val:.4f}',ha='center',va='bottom',
                  color='white',fontsize=12,fontweight='bold')

# Star on best
best_acc = np.argmax(accs)
best_f1  = np.argmax(f1s)
axes3[0].text(bars3a[best_acc].get_x()+bars3a[best_acc].get_width()/2,
              accs[best_acc]+4,'🏆',ha='center',fontsize=16)
axes3[1].text(bars3b[best_f1].get_x()+bars3b[best_f1].get_width()/2,
              f1s[best_f1]+0.04,'🏆',ha='center',fontsize=16)

fig3.suptitle('Model Performance Comparison',fontsize=16,
              fontweight='bold',color='white',y=1.02)
plt.tight_layout()
plt.savefig('data/fig3_comparison.png',dpi=150,bbox_inches='tight',
            facecolor='#0d1117')
print("✅ Fig 3 saved")

# ════════════════════════════════════════════════════════
# FIGURE 4 — Three Confusion Matrices
# ════════════════════════════════════════════════════════
fig4, axes4 = plt.subplots(1, 3, figsize=(24, 7))
fig4.patch.set_facecolor('#0d1117')

def plot_cm(ax, cm, labels, title, cmap):
    ax.set_facecolor('#161b22')
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    n = len(labels)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels,rotation=35,ha='right',
                       fontsize=8,color='white')
    ax.set_yticklabels(labels,fontsize=8,color='white')
    ax.set_xlabel('Predicted',color='white',fontsize=10)
    ax.set_ylabel('True',color='white',fontsize=10)
    ax.set_title(title,fontsize=11,fontweight='bold',color='white',pad=8)
    ax.tick_params(colors='white')
    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        color = 'white' if cm_norm[i,j]>0.5 else '#c9d1d9'
        ax.text(j,i,f'{cm_norm[i,j]:.0%}\n({cm[i,j]})',
                ha='center',va='center',color=color,fontsize=7)

# CM1 - PAMAP2
cm1 = confusion_matrix(ytp,pp)
plot_cm(axes4[0],cm1,list(PAMAP2_ACTIVITIES.values()),
        f'ResNet / PAMAP2\nAcc={acc_pamap:.2%}  F1={f1_pamap:.3f}',
        'Blues')

# CM2 - HARNet
cm2 = confusion_matrix(yph,ph)
plot_cm(axes4[1],cm2,list(PHYSIO_ACTIVITIES.values()),
        f'HARNet10 / PhysioNet\nAcc={acc_harnet:.2%}  F1={f1_harnet:.3f}',
        'Greens')

# CM3 - Fusion
cm3 = confusion_matrix(y_test,final_preds,labels=test_classes)
plot_cm(axes4[2],cm3,test_labels,
        f'Fusion Model (Subject-wise)\nAcc={acc_fusion:.2%}  F1={f1_fusion:.3f}',
        'Purples')

fig4.suptitle('Confusion Matrices — All Models',fontsize=16,
              fontweight='bold',color='white',y=1.02)
plt.tight_layout()
plt.savefig('data/fig4_confusion_matrices.png',dpi=150,
            bbox_inches='tight',facecolor='#0d1117')
print("✅ Fig 4 saved")

# ════════════════════════════════════════════════════════
# FIGURE 5 — Sample Accelerometer Signals
# ════════════════════════════════════════════════════════
fig5, axes5 = plt.subplots(2, 4, figsize=(20, 8))
fig5.patch.set_facecolor('#0d1117')

activities_to_show = [
    (X_pamap, y_pamap, 4, 'Walking',        '#388bfd', PAMAP2_ACTIVITIES),
    (X_pamap, y_pamap, 5, 'Running',         '#f78166', PAMAP2_ACTIVITIES),
    (X_pamap, y_pamap, 6, 'Cycling',         '#3fb950', PAMAP2_ACTIVITIES),
    (X_pamap, y_pamap, 1, 'Lying',           '#bc8cff', PAMAP2_ACTIVITIES),
    (X_phys,  y_phys,  1, 'Stair Climbing',  '#d29922', PHYSIO_ACTIVITIES),
    (X_phys,  y_phys,  2, 'Walking 6MWT',    '#58a6ff', PHYSIO_ACTIVITIES),
    (X_phys,  y_phys,  4, 'Timed Up & Go',   '#ff7b72', PHYSIO_ACTIVITIES),
    (X_phys,  y_phys,  5, 'Cycling (Phys)',  '#56d364', PHYSIO_ACTIVITIES),
]

time_axis = np.arange(300) / 30

for idx, (X, y, act_id, name, color, _) in enumerate(activities_to_show):
    row, col = idx // 4, idx % 4
    ax = axes5[row, col]
    ax.set_facecolor('#161b22')
    ax.spines['bottom'].set_color('#30363d')
    ax.spines['left'].set_color('#30363d')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    samples = np.where(y == act_id)[0]
    if len(samples) == 0:
        continue
    sample = X[samples[0]]
    ax.plot(time_axis, sample[:,0], color=color,
            lw=1.5, alpha=0.9, label='X')
    ax.plot(time_axis, sample[:,1], color=color,
            lw=1.0, alpha=0.5, linestyle='--', label='Y')
    ax.plot(time_axis, sample[:,2], color=color,
            lw=1.0, alpha=0.3, linestyle=':', label='Z')
    ax.set_title(name, color='white', fontsize=11, fontweight='bold')
    ax.tick_params(colors='white', labelsize=8)
    ax.set_xlabel('Time (s)', color='#8b949e', fontsize=8)
    ax.set_ylabel('Acceleration', color='#8b949e', fontsize=8)

fig5.suptitle('Raw Accelerometer Signals by Activity',
              fontsize=16, fontweight='bold', color='white', y=1.02)
plt.tight_layout()
plt.savefig('data/fig5_signals.png', dpi=150,
            bbox_inches='tight', facecolor='#0d1117')
print("✅ Fig 5 saved")

print("\n" + "="*50)
print("All 5 figures saved to data/ folder:")
print("  data/fig1_pipeline.png")
print("  data/fig2_datasets.png")
print("  data/fig3_comparison.png")
print("  data/fig4_confusion_matrices.png")
print("  data/fig5_signals.png")
print("="*50)
