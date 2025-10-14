
import os, argparse, json, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import classification_report
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime

from data import load_dataset
from model_tcn import TCN
from utils import class_weights_from_counts, save_json
from viz import save_curves, save_confusion
"""
python train_splitlog_ts_ws.py \
  --data-root ./data_set_v02 \
  --win 30 --stride 20 \
  --features xyconf --norm image \
  --batch 64 --epochs 50 \
  --split intra_session \
  --val-ratio 0.15 --test-ratio 0.15 \
  --out runs/exp1



"""
class NumpySeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = X; self.y = y
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        x = torch.from_numpy(self.X[i].transpose(1,0))  # (C,T)
        y = torch.tensor(self.y[i], dtype=torch.long)
        return x, y

def split_train_val_test(idx, y, val_ratio, test_ratio, seed):
    if test_ratio > 0:
        idx_tmp, idx_te = train_test_split(idx, test_size=test_ratio, random_state=seed, stratify=y)
        y_tmp = y[idx_tmp]
        rel_val = val_ratio / (1.0 - test_ratio)
        idx_tr, idx_va = train_test_split(idx_tmp, test_size=rel_val, random_state=seed, stratify=y_tmp)
        return idx_tr, idx_va, idx_te
    else:
        idx_tr, idx_va = train_test_split(idx, test_size=val_ratio, random_state=seed, stratify=y)
        return idx_tr, idx_va, None

def split_by_session(sessions, y, val_ratio, test_ratio, seed):
    sessions = np.asarray(sessions)
    uniq = np.unique(sessions)
    sess_to_label = {}
    for s in uniq:
        ys = y[sessions == s]
        sess_to_label[s] = int(ys[0])
    sess_label = np.array([sess_to_label[s] for s in uniq])
    _, counts = np.unique(sess_label, return_counts=True)
    need_strat = counts.min() >= 2
    rng = np.random.RandomState(seed)
    if need_strat:
        if test_ratio > 0:
            sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
            rest_idx, test_idx = next(sss1.split(uniq, sess_label))
            rest_sess, test_sess = uniq[rest_idx], uniq[test_idx]
            rest_labels = sess_label[rest_idx]
            rel_val = val_ratio / (1.0 - test_ratio)
            sss2 = StratifiedShuffleSplit(n_splits=1, test_size=rel_val, random_state=seed)
            tr_idx, va_idx = next(sss2.split(rest_sess, rest_labels))
            train_sess, val_sess = rest_sess[tr_idx], rest_sess[va_idx]
        else:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
            tr_idx, va_idx = next(sss.split(uniq, sess_label))
            train_sess, val_sess = uniq[tr_idx], uniq[va_idx]
            test_sess = np.array([], dtype=train_sess.dtype)
    else:
        print("[WARN] Some classes have <2 sessions; falling back to *unstratified* session split.")
        rng.shuffle(uniq)
        n_total = len(uniq)
        n_test = int(round(n_total * test_ratio))
        n_val  = int(round(n_total * val_ratio))
        test_sess = uniq[:n_test]
        val_sess  = uniq[n_test:n_test+n_val]
        train_sess= uniq[n_test+n_val:]
    tr_mask = np.isin(sessions, train_sess)
    va_mask = np.isin(sessions, val_sess)
    te_mask = np.isin(sessions, test_sess)
    idx_tr = np.where(tr_mask)[0]
    idx_va = np.where(va_mask)[0]
    idx_te = np.where(te_mask)[0] if test_ratio > 0 else None
    return idx_tr, idx_va, idx_te

def split_intra_session(sessions, val_ratio, test_ratio, seed):
    import hashlib
    sessions = np.asarray(sessions)
    uniq = np.unique(sessions)
    idx_tr_all, idx_va_all, idx_te_all = [], [], []
    for s in uniq:
        idx = np.where(sessions == s)[0].astype(int)
        h = int(hashlib.md5((str(seed)+str(s)).encode()).hexdigest(), 16) % (2**32)
        rng = np.random.RandomState(h)
        rng.shuffle(idx)
        n = len(idx)
        n_te = int(round(n * test_ratio))
        n_va = int(round(n * val_ratio))
        if n_te + n_va > n:
            overflow = n_te + n_va - n
            take = min(overflow, n_va); n_va -= take; overflow -= take
            if overflow > 0: n_te = max(0, n_te - overflow)
        te = idx[:n_te]
        va = idx[n_te:n_te+n_va]
        tr = idx[n_te+n_va:]
        if len(tr): idx_tr_all.append(tr)
        if len(va): idx_va_all.append(va)
        if len(te): idx_te_all.append(te)
    idx_tr = np.concatenate(idx_tr_all) if idx_tr_all else np.array([], dtype=int)
    idx_va = np.concatenate(idx_va_all) if idx_va_all else np.array([], dtype=int)
    idx_te = np.concatenate(idx_te_all) if idx_te_all else None
    return idx_tr, idx_va, idx_te

def counts_dict(y, classes):
    cnt = np.bincount(y, minlength=len(classes))
    total = int(cnt.sum())
    d = {classes[i]: {'count': int(cnt[i]), 'ratio': float(cnt[i]/total) if total>0 else 0.0} for i in range(len(classes))}
    return d, total

def split_summary(idx_tr, idx_va, idx_te, y, sessions, classes):
    def uniq_sessions(idxs):
        return int(len(np.unique(sessions[idxs]))) if idxs is not None and len(idxs)>0 else 0
    def part(idxs):
        if idxs is None: return {'size': 0, 'sessions': 0, 'labels': {}}
        d,t = counts_dict(y[idxs], classes)
        return {'size': int(len(idxs)), 'sessions': uniq_sessions(idxs), 'labels': d}
    return {'train': part(idx_tr), 'val': part(idx_va), 'test': part(idx_te) if idx_te is not None else None}

def save_split_logs(outdir, idx_tr, idx_va, idx_te, y, sessions, classes, split_mode, seed, val_ratio, test_ratio):
    os.makedirs(outdir, exist_ok=True)
    summ = split_summary(idx_tr, idx_va, idx_te, y, sessions, classes)
    meta = {'split': split_mode, 'seed': int(seed), 'val_ratio': float(val_ratio), 'test_ratio': float(test_ratio), 'classes': classes}
    obj = {'meta': meta, 'summary': summ}
    with open(os.path.join(outdir, 'split_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    np.savez(os.path.join(outdir, 'split_indices.npz'),
             idx_tr=np.asarray(idx_tr, dtype=np.int64),
             idx_va=np.asarray(idx_va, dtype=np.int64),
             idx_te=np.asarray(idx_te if idx_te is not None else [], dtype=np.int64))
    lines = []
    lines.append(f"[SPLIT] mode={split_mode} seed={seed} val={val_ratio} test={test_ratio}")
    for k in ['train','val','test']:
        part = summ[k] if k in summ else None
        if part is None:
            lines.append(f"  {k:>5s}: (none)")
            continue
        lines.append(f"  {k:>5s}: size={part['size']}, sessions={part['sessions']}")
        for cls, s in part['labels'].items():
            lines.append(f"        - {cls:>15s}: {s['count']:4d} ({s['ratio']*100:5.1f}%)")
    with open(os.path.join(outdir, 'split_summary.txt'), 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print("\n".join(lines))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', type=str, required=True)
    ap.add_argument('--win', type=int, default=60)
    ap.add_argument('--stride', type=int, default=20)
    ap.add_argument('--features', type=str, default='xyconf', choices=['xy','xyconf'])
    ap.add_argument('--norm', type=str, default='image', choices=['image','center_hips'])
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--levels', type=int, default=4)
    ap.add_argument('--channels', type=int, default=128)
    ap.add_argument('--kernel', type=int, default=3)
    ap.add_argument('--val-ratio', type=float, default=0.15)
    ap.add_argument('--test-ratio', type=float, default=0.15)
    ap.add_argument('--split', type=str, default='intra_session', choices=['within_session','by_session','intra_session'])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--class-weights', type=str, default='auto', choices=['none','auto'])
    ap.add_argument('--out', type=str, default='runs/exp1')
    ap.add_argument('--export-best-to', type=str, default='/home/ubuntu/workspace/AI_project/tcn_infer_package', help="Also copy best.pt here (set '' to disable)")
    args = ap.parse_args()

    # ----- timestamped + win/stride in output dir name -----
    ts = datetime.now().strftime('%Y%m%d_%H_%M_%S')
    base_out = args.out.rstrip('/')
    args.out = f"{base_out}_w{args.win}_s{args.stride}_{ts}"
    print(f"[OUT] saving to: {args.out}")

    assert 0 <= args.val_ratio < 1 and 0 <= args.test_ratio < 1 and args.val_ratio + args.test_ratio < 1, 'val_ratio+test_ratio must be < 1.0'

    os.makedirs(args.out, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print('[Load] building sequences...')
    X, y, classes, sessions = load_dataset(args.data_root, win=args.win, stride=args.stride, features=args.features, norm=args.norm)
    print(f'X={X.shape} y={y.shape} classes={classes}')

    C = X.shape[2]
    idx = np.arange(len(X))

    if args.split == 'by_session':
        idx_tr, idx_va, idx_te = split_by_session(sessions, y, args.val_ratio, args.test_ratio, args.seed)
    elif args.split == 'intra_session':
        idx_tr, idx_va, idx_te = split_intra_session(sessions, args.val_ratio, args.test_ratio, args.seed)
    else:
        idx_tr, idx_va, idx_te = split_train_val_test(idx, y, args.val_ratio, args.test_ratio, args.seed)

    save_split_logs(args.out, idx_tr, idx_va, idx_te, y, sessions, classes,
                    split_mode=args.split, seed=args.seed, val_ratio=args.val_ratio, test_ratio=args.test_ratio)

    Xtr, ytr = X[idx_tr], y[idx_tr]
    Xva, yva = X[idx_va], y[idx_va]
    Xte, yte = (X[idx_te], y[idx_te]) if idx_te is not None else (None, None)

    # label ratio json
    def counts_dict_simple(y, classes):
        cnt = np.bincount(y, minlength=len(classes))
        total = int(cnt.sum())
        d = {classes[i]: {'count': int(cnt[i]), 'ratio': float(cnt[i]/total) if total>0 else 0.0} for i in range(len(classes))}
        return d, total
    train_dict, train_total = counts_dict_simple(ytr, classes)
    val_dict, val_total = counts_dict_simple(yva, classes)
    out_json = {'train': train_dict, 'train_total': train_total, 'val': val_dict, 'val_total': val_total, 'idx2label': classes}
    if Xte is not None:
        test_dict, test_total = counts_dict_simple(yte, classes)
        out_json.update({'test': test_dict, 'test_total': test_total})
    save_json(out_json, os.path.join(args.out, 'label_ratio.json'))

    dtr = NumpySeqDataset(Xtr, ytr)
    dva = NumpySeqDataset(Xva, yva)
    loader_tr = DataLoader(dtr, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True)
    loader_va = DataLoader(dva, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

    model = TCN(input_channels=C, num_classes=len(classes),
                num_levels=args.levels, n_channels=args.channels, kernel_size=args.kernel, dropout=args.dropout).to(device)

    if args.class_weights == 'auto':
        counts = np.bincount(ytr, minlength=len(classes))
        weight = class_weights_from_counts(counts).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight)
    else:
        criterion = nn.CrossEntropyLoss()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1,args.epochs))

    writer = SummaryWriter(log_dir=os.path.join(args.out, 'tb'))
    save_json({'args': vars(args), 'classes': classes}, os.path.join(args.out, 'hparams.json'))
    writer.add_scalar('data/num_train', len(dtr), 0)
    writer.add_scalar('data/num_val', len(dva), 0)
    if Xte is not None: writer.add_scalar('data/num_test', len(Xte), 0)

    history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_acc, best_path = -1.0, os.path.join(args.out, 'best.pt')
    best_state = None

    for epoch in range(1, args.epochs+1):
        model.train(); tr_loss = 0.0
        for x,yb in tqdm(loader_tr, desc=f'epoch {epoch}/{args.epochs} [train]'):
            x,yb = x.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, yb)
            loss.backward(); opt.step()
            tr_loss += float(loss.item())*len(x)
        tr_loss /= max(1,len(dtr))

        model.eval(); y_true=[]; y_pred=[]; va_loss=0.0
        with torch.no_grad():
            for x,yb in tqdm(loader_va, desc=f'epoch {epoch}/{args.epochs} [val]'):
                x,yb = x.to(device), yb.to(device)
                logits = model(x)
                loss = criterion(logits, yb)
                va_loss += float(loss.item())*len(x)
                pred = logits.argmax(dim=1).cpu().numpy()
                y_pred.extend(pred.tolist()); y_true.extend(yb.cpu().numpy().tolist())
        va_loss /= max(1,len(dva))
        acc = float((np.array(y_true)==np.array(y_pred)).mean())

        writer.add_scalar('loss/train', tr_loss, epoch)
        writer.add_scalar('loss/val', va_loss, epoch)
        writer.add_scalar('acc/val', acc, epoch)
        writer.add_scalar('lr', opt.param_groups[0]['lr'], epoch)

        history['epoch'].append(epoch); history['train_loss'].append(tr_loss)
        history['val_loss'].append(va_loss); history['val_acc'].append(acc)

        if acc > best_acc:
            best_acc = acc
            best_state = {'model': model.state_dict(), 'classes': classes, 'input_channels': C, 'hparams': vars(args)}
            torch.save(best_state, best_path)
            # --- export copy of best.pt to user-specified path ---
            exp_dir = getattr(args, 'export_best_to', None)
            if exp_dir is not None and str(exp_dir) != '':
                try:
                    import os
                    os.makedirs(exp_dir, exist_ok=True)
                    exp_path = os.path.join(exp_dir, 'best.pt')
                    torch.save(best_state, exp_path)
                    print(f"[EXPORT] best.pt copied to: {exp_path}")
                except Exception as e:
                    print(f"[WARN] failed to export best.pt to {exp_dir}: {e}")

        sched.step()

    last_path = os.path.join(args.out, 'last.pt')
    torch.save({'model': model.state_dict(),'classes': classes,'input_channels': C,'hparams': vars(args)}, last_path)

    save_curves(history, args.out)

    if best_state is not None:
        model.load_state_dict(best_state['model'])
    from sklearn.metrics import classification_report as _cr
    y_true, y_pred = [], []
    with torch.no_grad():
        for x,yb in DataLoader(NumpySeqDataset(Xva,yva), batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True):
            x,yb = x.to(device), yb.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1).cpu().numpy()
            y_pred.extend(pred.tolist()); y_true.extend(yb.cpu().numpy().tolist())
    rep = _cr(y_true, y_pred, target_names=classes, output_dict=True, digits=6)
    save_confusion(y_true, y_pred, classes, os.path.join(args.out, 'confusion_val.png'))
    with open(os.path.join(args.out, 'report_val.json'), 'w', encoding='utf-8') as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out, 'report_val.txt'), 'w', encoding='utf-8') as f:
        f.write(_cr(y_true, y_pred, target_names=classes, digits=6))

    if idx_te is not None and len(idx_te)>0:
        dte = NumpySeqDataset(Xte, yte)
        loader_te = DataLoader(dte, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)
        y_true_te, y_pred_te = [], []
        with torch.no_grad():
            for x,yb in loader_te:
                x,yb = x.to(device), yb.to(device)
                logits = model(x)
                pred = logits.argmax(dim=1).cpu().numpy()
                y_pred_te.extend(pred.tolist()); y_true_te.extend(yb.cpu().numpy().tolist())
        rep_te = _cr(y_true_te, y_pred_te, target_names=classes, output_dict=True, digits=6)
        save_confusion(y_true_te, y_pred_te, classes, os.path.join(args.out, 'confusion_test.png'))
        with open(os.path.join(args.out, 'report_test.json'), 'w', encoding='utf-8') as f:
            json.dump(rep_te, f, ensure_ascii=False, indent=2)
        with open(os.path.join(args.out, 'report_test.txt'), 'w', encoding='utf-8') as f:
            f.write(_cr(y_true_te, y_pred_te, target_names=classes, digits=6))

    print('\\n[VAL REPORT]')
    print(_cr(y_true, y_pred, target_names=classes, digits=6))
    if idx_te is not None and len(idx_te)>0:
        print('\\n[TEST REPORT]')
        from sklearn.metrics import classification_report as _cr2
        print(_cr2(y_true_te, y_pred_te, target_names=classes, digits=6))

    print(f'[DONE] best_acc={best_acc:.4f}  saved: {best_path}  logs: {args.out}')

if __name__ == '__main__':
    main()
