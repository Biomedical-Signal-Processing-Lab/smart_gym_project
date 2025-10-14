
import os, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def save_curves(history, outdir):
    os.makedirs(outdir, exist_ok=True)
    fig = plt.figure()
    plt.plot(history['epoch'], history['train_loss'], label='train')
    plt.plot(history['epoch'], history['val_loss'], label='val')
    plt.xlabel('epoch'); plt.ylabel('loss'); plt.legend(); plt.grid(True)
    fig.savefig(os.path.join(outdir, 'loss_curve.png'), dpi=150, bbox_inches='tight'); plt.close(fig)
    if 'val_acc' in history:
        fig = plt.figure()
        plt.plot(history['epoch'], history['val_acc'], label='val_acc')
        plt.xlabel('epoch'); plt.ylabel('acc'); plt.legend(); plt.grid(True)
        fig.savefig(os.path.join(outdir, 'acc_curve.png'), dpi=150, bbox_inches='tight'); plt.close(fig)

def save_confusion(y_true, y_pred, class_names, outpath):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(6,6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    fig.tight_layout(); fig.savefig(outpath, dpi=150); plt.close(fig)
