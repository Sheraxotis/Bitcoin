"""Model-independent direction metrics and compact reports."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, log_loss,
                             brier_score_loss, confusion_matrix, precision_score, recall_score, f1_score,
                             roc_curve, roc_auc_score)


def scores(y, predicted, probability):
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return dict(n=len(y), accuracy=accuracy_score(y, predicted),
                balanced_accuracy=balanced_accuracy_score(y, predicted),
                precision=precision_score(y, predicted, zero_division=0),
                recall=recall_score(y, predicted, zero_division=0), f1=f1_score(y, predicted, zero_division=0),
                log_loss=log_loss(y, probability, labels=[0, 1]), brier_score=brier_score_loss(y, probability),
                roc_auc=roc_auc_score(y, probability) if pd.Series(y).nunique()==2 else float('nan'),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def plot_roc(pred, names, config, out, results=None):
    if results is None:
        results = pd.DataFrame([dict(model=name, **scores(pred.actual_class, pred[name+'_class'],
                                                         pred[name+'_probability'])) for name in names])
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for name in names:
        if pred.actual_class.nunique()==2:
            fpr, tpr, _ = roc_curve(pred.actual_class, pred[name+'_probability'])
            auc = results.loc[results.model==name, 'roc_auc'].iloc[0]
            ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=.5, label='Chance (AUC=0.5)')
    if pred.actual_class.nunique()!=2:
        ax.text(.1, .5, 'ROC undefined: only one actual class')
    ax.set(xlabel='False positive rate (FPR)', ylabel='True positive rate (TPR)',
           xlim=(0, 1), ylim=(0, 1), title=f"BTCUSDT {config['interval']} out-of-sample ROC")
    ax.legend(); ax.grid(alpha=.3)
    fig.savefig(out/'roc_curve.png', dpi=160)
    plt.close(fig)


def write_reports(pred, names, config, out):
    results = pd.DataFrame([dict(model=name, **scores(pred.actual_class, pred[name+'_class'],
                                                     pred[name+'_probability'])) for name in names])
    results.to_csv(out/'overall_metrics.csv', index=False)
    plot_roc(pred, names, config, out, results)
    lines = [f"# {config['interval']} classification", '',
             f"UTC evaluation: [{config['evaluation_start']}, {config['evaluation_end']}); {len(pred)} shared rows.",
             f"Shared features: {', '.join(config['feature_columns'])}.",
             f"Training window: {config['window_days']} calendar days; refit every {config['update_hours']} hours.",
             'Target: next-bar return > 0. Only labels known at the training cutoff are used.',
             'Every model uses the same features, training splits and prediction rows.', '',
             '| Model | Accuracy | Balanced accuracy | Precision | Recall | F1 | Log loss | Brier | ROC AUC |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in results.itertuples():
        lines.append(f'| {r.model} | {r.accuracy:.2%} | {r.balanced_accuracy:.2%} | {r.precision:.2%} | '
                     f'{r.recall:.2%} | {r.f1:.2%} | {r.log_loss:.6f} | {r.brier_score:.6f} | {r.roc_auc:.6f} |')
    lines += ['', 'Zero predicts non-up with probability 0; majority uses the current training up-rate.',
              'Log loss clips exact zero probabilities; beating that baseline does not establish useful probability quality.',
              'Accuracy, precision, recall and F1 are higher-is-better; log loss and Brier are lower-is-better.',
              'ROC sweeps probability thresholds; AUC measures ranking, not probability calibration or trading profit.',
              'AUC is undefined for a single actual class. Trading signals remain at the configured fixed threshold 0.5.',
              'Probability quality is not trading profitability. This repeatedly inspected history is exploratory.', '']
    (out/'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    return results
