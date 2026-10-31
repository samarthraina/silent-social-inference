"""Plot both predeclared endpoints for both models; include nulls and missing-score bounds."""
import argparse
import json
from pathlib import Path


def plot(summary,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    names=['history_gap_reduction','history_vs_control_specificity']
    labels=['History-gap reduction','History vs content-control specificity']
    if {m['model'] for m in summary['models']}!={'llama','qwen'}:raise ValueError('Both models are required')
    fig,axes=plt.subplots(1,2,figsize=(11,3.6),sharey=True)
    for ax,model in zip(axes,summary['models']):
        for y,name in enumerate(names):
            r=model['primary'][name];value=r['mean_points'];ci=r['ci'];lo,hi=r['planned_task_missing_bounds']
            ax.plot([lo,hi],[y+.13,y+.13],ls='--',lw=2,color='#A87538')
            if value is not None:
                ax.plot(ci,[y,y],lw=2,color='#176B87');ax.plot(value,y,'o',color='#176B87')
            ax.text(.98,y-.13,f"n={r['complete_tasks']}/{r['planned_tasks']}",ha='right',va='bottom',transform=ax.get_yaxis_transform(),fontsize=8)
        ax.axvline(0,lw=.8,color='#667788');ax.set_title(model['model'].capitalize())
        ax.set_yticks(range(2),labels);ax.set_ylim(1.6,-.5);ax.set_xlabel('Score points (positive = planned direction)')
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.15)
    title='Held-out confirmation | generated-reader history access'
    if summary.get('synthetic'):title='SYNTHETIC RENDER CHECK - NOT RESEARCH DATA'
    fig.suptitle(title)
    fig.legend(handles=[Line2D([0],[0],color='#176B87',marker='o',label='Complete-case mean and 98.75% family-adjusted bootstrap CI'),
                        Line2D([0],[0],color='#A87538',ls='--',label='All-task worst-case missing-score bounds (not CI)')],loc='lower center',ncol=1,frameon=False,fontsize=8)
    fig.tight_layout(rect=[0,.18,1,.9]);out.mkdir(parents=True,exist_ok=False)
    fig.savefig(out/'confirmation_effects.png',dpi=240);fig.savefig(out/'confirmation_effects.svg');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--summary',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();plot(json.loads(a.summary.read_text()),a.out)
