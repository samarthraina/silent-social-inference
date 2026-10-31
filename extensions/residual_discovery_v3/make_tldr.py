import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

OUT=Path(__file__).resolve().parent/'analysis';OUT.mkdir(exist_ok=True)
DATA=json.loads((OUT/'audit_and_results.json').read_text())
NAVY='#163047';TEAL='#087E8B';BLUE='#2166AC';ORANGE='#B35806';MUTED='#526575';PALE='#EFF6F8'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
    'axes.spines.right':False,'axes.unicode_minus':False,'savefig.facecolor':'white'})

# Figure 1: paired behavioral contrasts; no pooling with the earlier 72-task study.
fig,ax=plt.subplots(figsize=(7.2,1.85))
for y,model in enumerate(['llama','qwen']):
    b=DATA['models'][model]['baseline'];v=b['gap_mean'];lo,hi=b['gap_ci95']
    ax.barh(y,v,height=.43,color=TEAL,alpha=.18)
    ax.errorbar(v,y,xerr=[[v-lo],[hi-v]],fmt='o',color=TEAL,capsize=4,markersize=6,lw=1.7)
    ax.text(hi+.10,y,f'+{v:.2f}',va='center',fontweight='bold',color=NAVY,fontsize=11)
ax.set_yticks([0,1],['Llama 3.1 8B','Qwen3 8B']);ax.invert_yaxis()
ax.set_xlim(0,3.7);ax.set_ylim(1.55,-.55)
ax.set_xlabel('Positive-history score minus negative-history score (points)')
ax.set_xticks([0,1,2,3]);ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
fig.tight_layout(pad=.8);fig.savefig(OUT/'figure_1_behavior.png',dpi=300);plt.close(fig)

# Figure 2: every searched cell, shown in original score units.
layers=[4,8,12,16,20,24,28,32];spans=['history_end','answer','final']
fig,axes=plt.subplots(2,2,figsize=(7.2,4.7))
for row,model in enumerate(['llama','qwen']):
    for col,source in enumerate(['self_high_10','self_low_2']):
        ax=axes[row,col]
        cells=DATA['models'][model]['cells']
        lookup={(c['layer'],c['span']):c for c in cells if c['source']==source}
        arr=np.array([[lookup[l,s]['mean_points'] for s in spans] for l in layers])
        im=ax.imshow(arr,cmap='RdBu',vmin=-.5,vmax=.5,aspect='auto')
        for i,l in enumerate(layers):
            for j,s in enumerate(spans):
                c=lookup[l,s];v=c['mean_points'];label=f'{v:.2f}'
                if abs(v)<.00001:label='0.00'
                if c['n_missing']:label+='*'
                ax.text(j,i,label,ha='center',va='center',fontsize=8,
                        color='white' if abs(v)>.31 else NAVY)
        ax.set_yticks(range(8),layers);ax.set_xticks(range(3),['History end','Answer span','Final input'])
        ax.tick_params(axis='both',length=0,labelsize=8)
        ax.set_xticks(np.arange(-.5,3,1),minor=True);ax.set_yticks(np.arange(-.5,8,1),minor=True)
        ax.grid(which='minor',color='white',lw=1);ax.tick_params(which='minor',length=0)
        title=('Llama' if model=='llama' else 'Qwen')+' | '+('High -> low' if col==0 else 'Low -> high')
        ax.set_title(title,fontsize=10,fontweight='bold',color=NAVY,pad=8)
        if col==0:ax.set_ylabel('Post-block layer',fontsize=8)
        for spine in ax.spines.values():spine.set_visible(False)
fig.subplots_adjust(left=.09,right=.98,bottom=.17,top=.92,wspace=.32,hspace=.49)
cax=fig.add_axes([.27,.085,.49,.025]);cb=fig.colorbar(im,cax=cax,orientation='horizontal',ticks=[-.5,-.25,0,.25,.5])
cb.ax.tick_params(labelsize=8);cb.set_label('Mean movement toward donor score (points)',fontsize=8,labelpad=3)
fig.savefig(OUT/'figure_2_patch_map.png',dpi=300);plt.close(fig)

fontdir=Path(matplotlib.get_data_path())/'fonts/ttf'
pdfmetrics.registerFont(TTFont('DV',str(fontdir/'DejaVuSans.ttf')))
pdfmetrics.registerFont(TTFont('DV-Bold',str(fontdir/'DejaVuSans-Bold.ttf')))
pdfmetrics.registerFont(TTFont('DV-Oblique',str(fontdir/'DejaVuSans-Oblique.ttf')))
pdfmetrics.registerFontFamily('DV',normal='DV',bold='DV-Bold',italic='DV-Oblique',boldItalic='DV-Bold')
W,H=A4;M=43;CW=W-2*M
styles={
 'kicker':ParagraphStyle('k',fontName='DV-Bold',fontSize=8,leading=11,textColor=colors.HexColor(TEAL),spaceAfter=7),
 'title':ParagraphStyle('t',fontName='DV-Bold',fontSize=23,leading=28,textColor=colors.HexColor(NAVY),spaceAfter=9),
 'sub':ParagraphStyle('s',fontName='DV',fontSize=10,leading=14,textColor=colors.HexColor(MUTED),spaceAfter=10),
 'h':ParagraphStyle('h',fontName='DV-Bold',fontSize=12,leading=16,textColor=colors.HexColor(NAVY),spaceBefore=9,spaceAfter=5),
 'body':ParagraphStyle('b',fontName='DV',fontSize=9.7,leading=13.5,textColor=colors.HexColor(NAVY),spaceAfter=7),
 'small':ParagraphStyle('sm',fontName='DV',fontSize=8,leading=10.5,textColor=colors.HexColor(MUTED),spaceAfter=5),
 'cell':ParagraphStyle('c',fontName='DV',fontSize=8.6,leading=11.5,textColor=colors.HexColor(NAVY)),
 'head':ParagraphStyle('ch',fontName='DV-Bold',fontSize=8.4,leading=11,textColor=colors.white),
}
def P(text,style='body'):return Paragraph(text,styles[style])
def T(rows,widths,header=True):
    data=[[P(str(v),'head' if header and i==0 else 'cell') for v in row] for i,row in enumerate(rows)]
    table=Table(data,colWidths=widths,hAlign='LEFT')
    cmds=[('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),8),
          ('RIGHTPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),7),
          ('BOTTOMPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,-1),(-1,-1),.4,colors.HexColor('#CAD8DE'))]
    if header:cmds+=[('BACKGROUND',(0,0),(-1,0),colors.HexColor(NAVY)),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#F0F5F7'),colors.white])]
    table.setStyle(TableStyle(cmds));return table
def box(text):
    t=Table([[P(text)]],colWidths=[CW]);t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,-1),colors.HexColor(PALE)),('BOX',(0,0),(-1,-1),.6,colors.HexColor('#BCD6DC')),
        ('LEFTPADDING',(0,0),(-1,-1),11),('RIGHTPADDING',(0,0),(-1,-1),11),
        ('TOPPADDING',(0,0),(-1,-1),10),('BOTTOMPADDING',(0,0),(-1,-1),5)]));return t
def image(path):
    from PIL import Image as PILImage
    w,h=PILImage.open(path).size
    return Image(str(path),width=CW,height=CW*h/w)


story=[P('SILENT INFLUENCE | COMPLETED DISCOVERY | 31 OCT 2026','kicker'),
 P('Where is the history effect carried?','title'),
 P('A brief summary of the residual-state experiments','sub'),
 box('<b>Main finding:</b> positive versus negative history strongly changes scores. Replacing one tested internal state usually does not change the score. We have small leads, but no confirmed mechanism on unseen tasks.'),
 P('What we did','h'),
 P('Llama 3.1 8B and Qwen3 8B judged the same frozen answers under a reported prior rating of 10/10 or 2/10. Each model used the same 18 discovery tasks: six each from MT-Bench, Arena-Hard and WildBench.'),
 P('We copied internal prompt states between the two histories, in both directions. We tested 24 sites per model: eight layers (4 through 32, every four layers) at three locations: the last history token, the answer tokens, and the final prompt token. The model then generated its own evaluation and numeric score.'),
 P('History still changes the score','h'),image(OUT/'figure_1_behavior.png'),
 P('Mean paired score gap on a 0-10 scale. Bars show stratified 95% bootstrap confidence intervals; 18 tasks per model. All 18 tasks showed a positive gap in both models. These are the new discovery tasks, not the earlier 72-task experiment.','small'),
 T([['Model','10/10 history','2/10 history','Gap'],['Llama 3.1 8B','9.33','7.22','+2.11'],['Qwen3 8B','9.69','7.22','+2.47']],[CW*.30,CW*.24,CW*.24,CW*.22]),
 P('What this tells us','h'),
 P('The behavioral effect is clear. The intervention asks a narrower question: can a particular prompt state transfer it? A weak patch does not mean history has no internal effect. Information may be distributed, or the model may read history again while generating.'),
 P('Did the checks pass?','h'),
 P('Yes: all 3,528 planned discovery records are present. All 1,728 same-state identity checks and 24 embedding-recovery positive controls passed. Five outputs were unusable (four hit the token limit and one could not be parsed); they remain missing, not zero. Saved effects were reproduced from the raw records.'),
PageBreak(),
 P('WHAT THE INTERNAL INTERVENTIONS SHOW','kicker'),
 P('Internal replacements: small effects','title'),
 image(OUT/'figure_2_patch_map.png'),
 P('Each cell is the average score movement toward the donor history after replacing one site. Blue means movement in the expected direction; red means the opposite. H to L copies high-history states into the low-history run; L to H reverses this. An asterisk marks a cell with a missing usable score.','small'),
 P('<b>No score change in 79.5% of usable Llama patches and 89.4% of usable Qwen patches.</b> This compares each patched score with its matching unpatched baseline.'),
 T([['Exploratory lead','High to low','Low to high'],['Llama: layer 12, last history token','+0.31 points','+0.24 points'],['Qwen: layer 16, last history token','+0.39 points','+0.14 points']],[CW*.52,CW*.24,CW*.24]),
 P('Leads were chosen after searching the 24 sites, using the smaller of the two directional means. Each direction has 18 usable tasks except Llama low-to-high (17). Median movement is zero in all four lead comparisons. Their exploratory 98.75% intervals reach or cross zero and do not correct for the site search. These are hypotheses, not validated discoveries.','small'),
 P('What can we say in the rebuttal?','h'),
 box('A broader, controlled residual-state search confirms the behavioral sensitivity but does not establish a strong single-site carrier of the effect during free generation. Our discovery results constrain the mechanism claim; confirmation on jointly audited unseen tasks remains future work.'),
 P('Next, as separate follow-ups: block history reading only during generated tokens; combine that with prompt-state replacement; then compare normal and fixed-rationale scoring. These experiments have not been run here.','small')]
def footer(c,doc):
 c.setFont('DV',8);c.setFillColor(colors.HexColor(MUTED))
 c.drawString(M,25,'Residual discovery v3 | frozen answers | free-generation scores')
 c.drawRightString(W-M,25,str(doc.page))
path=Path(__file__).resolve().parent/'residual_discovery_TLDR.pdf'
SimpleDocTemplate(str(path),title='Residual discovery v3: TLDR',author='Silent Influence project',pagesize=A4,rightMargin=M,leftMargin=M,topMargin=35,bottomMargin=42).build(story,onFirstPage=footer,onLaterPages=footer)
print(path)
