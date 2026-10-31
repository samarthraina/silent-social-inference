"""Plot all predeclared outcomes, including nulls and planned-task missing-score bounds."""
import argparse
import json
from pathlib import Path

LABELS={
 'history_gap_reduction':'History-gap reduction',
 'history_vs_control_specificity':'History vs content-control specificity',
 'patch_block_interaction':'Patch x blocking interaction',
 'patch_interaction_vs_l2':'Interaction relative to L2',
 'neutral_rationale_gap_reduction':'Generic vs neutral-rationale gap',
 'rationale_donor_effect':'High vs low rationale carryover',
}

def plot(summary,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    models=summary['models']
    if len(models)!=2:raise ValueError('Both model summaries are required')
    fig,axes=plt.subplots(1,2,figsize=(12,5.7),sharey=True)
    for ax,m in zip(axes,models):
        for y,key in enumerate(LABELS):
            r=m['primary'][key];mean=r['complete_case_mean_points'];ci=r['ci_family_adjusted']
            lo,hi=r['planned_task_mean_bounds']
            ax.plot([lo,hi],[y+.12,y+.12],color='#A87538',lw=2,ls='--',alpha=.85)
            if mean is not None:
                # Percentile intervals need not bracket the original sample mean.
                ax.plot(ci,[y,y],color='#176B87',lw=2)
                ax.plot(mean,y,'o',color='#176B87',markersize=5)
            ax.text(.98,y-.15,f"n={r['complete_tasks']}/{r['planned_tasks']}",
                    transform=ax.get_yaxis_transform(),ha='right',va='bottom',fontsize=8,color='#526575')
        ax.axvline(0,color='#526575',lw=.8)
        ax.set_title(m['model'].capitalize(),fontweight='bold')
        ax.set_yticks(range(len(LABELS)),list(LABELS.values()));ax.set_ylim(len(LABELS)-.5,-.5)
        ax.set_xlabel('Score points (positive = planned direction)')
        ax.grid(axis='x',alpha=.15);ax.spines[['top','right']].set_visible(False)
    status='HELD-OUT VALIDATION' if all(m['confirmatory'] for m in models) else 'EXPLORATORY PILOT'
    if summary.get('synthetic'):status='SYNTHETIC RENDER TEST - NOT EXPERIMENT DATA'
    fig.suptitle(status+' | generated-reader and rationale-prefix interventions',fontsize=12)
    fig.legend(handles=[Line2D([0],[0],color='#176B87',marker='o',label='Complete-case mean and 99.5833% family-adjusted bootstrap CI'),
                        Line2D([0],[0],color='#A87538',ls='--',label='All-task worst-case missing-score bounds (not a confidence interval)')],
               loc='lower center',fontsize=9,ncol=1,frameon=False)
    fig.tight_layout(rect=[0,.11,1,.94]);out.mkdir(parents=True,exist_ok=False)
    fig.savefig(out/'primary_effects.png',dpi=240);fig.savefig(out/'primary_effects.svg');plt.close(fig)
    print('Figures:',out)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--summary',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    a=p.parse_args();plot(json.loads(a.summary.read_text()),a.out)
