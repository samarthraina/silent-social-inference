from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import argparse
ap=argparse.ArgumentParser(description='Plot audited paired results without rerunning models.')
ap.add_argument('--analysis-dir',type=Path,required=True)
args=ap.parse_args()
out=args.analysis_dir
audit=json.loads((out/'integrity_and_paired_results.json').read_text());summary=json.loads((out/'recomputed_summary.json').read_text())
lookup={m['model']:m for m in summary['models']}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False})
fig,axes=plt.subplots(1,2,figsize=(10,4.0),sharey=True)
order=['native_baseline','baseline','control_block','generated_block'];labels=['Native\nbaseline','Matched\nbaseline','Content\ncontrol','History\nblocked']
for ax,name in zip(axes,['llama','qwen']):
 vals=[audit['models'][name]['paired_arm_means'][a]['gap'] for a in order]
 ax.bar(range(4),vals,color=['#99AAB5','#426D8A','#B7A489','#087E8B'],width=.62)
 for x,y in enumerate(vals):ax.text(x,y+.07,f'{y:.2f}',ha='center',fontsize=10)
 ax.set_xticks(range(4),labels);ax.set_ylim(0,3.9);ax.set_title(name.capitalize()+' | '+f"{audit['models'][name]['paired_gap_reduction_percent']:.0f}% gap reduction",fontweight='bold')
 ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
axes[0].set_ylabel('Mean positive-minus-negative history score gap')
fig.suptitle('Held-out confirmation: direct history access during generation',fontweight='bold')
fig.text(.5,.01,'Descriptive means on the same 43 complete task pairs per model; 44 planned. Intervention comparisons use the matched baseline.',ha='center',fontsize=8,color='#526575')
fig.tight_layout(rect=[0,.08,1,.92]);fig.savefig(out/'paired_score_gaps.png',dpi=260);fig.savefig(out/'paired_score_gaps.svg');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10.6,3.7),sharey=True)
keys=['history_gap_reduction','history_vs_control_specificity'];labels=['Gap reduction vs baseline','History vs content specificity']
for ax,name in zip(axes,['llama','qwen']):
 for y,key in enumerate(keys):
  r=lookup[name]['primary'][key];ci=r['ci'];b=r['planned_task_missing_bounds']
  ax.plot(ci,[y,y],color='#176B87',lw=2);ax.plot(r['mean_points'],y,'o',color='#176B87')
  ax.plot(b,[y+.14,y+.14],ls='--',lw=2,color='#A87538')
 ax.set_title(name.capitalize(),fontweight='bold');ax.axvline(0,color='#526575',lw=.8)
 ax.set_yticks([0,1],labels);ax.set_ylim(1.5,-.4);ax.grid(axis='x',alpha=.15);ax.set_xlabel('Score points (positive = predeclared direction)')
fig.suptitle('Both predeclared confirmation checks are positive in both models',fontweight='bold')
fig.legend(handles=[Line2D([0],[0],color='#176B87',marker='o',label='Paired mean and 98.75% family-adjusted bootstrap interval'),Line2D([0],[0],color='#A87538',ls='--',label='All-44-task worst-case missing-score bounds (not a confidence interval)')],loc='lower center',ncol=1,frameon=False,fontsize=8)
fig.tight_layout(rect=[0,.16,1,.9]);fig.savefig(out/'primary_effects_and_missing_bounds.png',dpi=260);fig.savefig(out/'primary_effects_and_missing_bounds.svg');plt.close(fig)
print('Created paired-gap and primary-effect figures.')
