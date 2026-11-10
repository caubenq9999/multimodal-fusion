from pathlib import Path
import re
import zipfile

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / 'overleaf_query_adaptive_fusion'
OUT.mkdir(exist_ok=True)
source = (BASE / 'PAPER_DRAFT_Sections_1_2_4.md').read_text(encoding='utf-8')
dataset = r'''### 4.1 Dataset

**Budgeted retrieval pilot.** We select MSR-VTT [6] as the source and plan a custom sample of 500 unique videos, rather than processing the full dataset. This sample is intended to test the fusion hypothesis within a limited computational budget. It is not the standard 1K-A protocol, and its scores will not be compared numerically with full-benchmark results. MSR-VTT provides reference descriptions supporting query-to-video relevance; actual video availability and annotation integrity will be verified before sampling.

**Sampling and eligibility.** Before any retrieval scores are inspected, we build an eligible manifest of available, decodable videos with valid IDs and at least one nonempty reference description. We report the source release, eligible population, exclusions, and download failures. From the sorted manifest, videos are sampled without replacement using seed 42. When source category labels are available, proportional stratification uses largest-remainder quotas; otherwise uniform random sampling is used and the realized category distribution is reported. Semantic query categories used for fusion are distinct from source dataset categories. Clips are not selected according to branch performance, caption quality, or apparent retrieval difficulty.

**Video-disjoint supervision.** The sampled video IDs are assigned to 300 training, 100 validation, and 100 test groups, preserving source-category proportions when feasible. All descriptions of a video remain in the same supervision partition. Within each video, at most five distinct reference descriptions are sampled using a fixed seed before model fitting. This yields at most 1,500 training, 500 validation, and 500 test queries; actual counts are reported. The descriptions are query annotations only and are excluded from the generated-caption index. Duplicate text associated with different videos is flagged for relevance review and excluded from query sampling if ambiguous; it is not automatically labeled as a negative for another matching video.

**Shared candidate collection.** Every query ranks the same 500 candidate videos. Training, validation, and test refer to query-supervision partitions; they do not partition the retrieval index. Frozen visual and caption features for all candidates are computed independently of the query annotations. Test-positive videos can therefore appear as distractors for other query groups, but test queries and their relevance labels never enter gate training or model selection. This is a fixed-collection pilot, not a claim of generalization to an unseen candidate collection. Learned video projections or encoders are outside this protocol and would require a separate candidate-disjoint design.

**Index construction.** We initially sample four frames per video at relative positions (i + 0.5)/4 for i = 0,1,2,3 and generate one caption per frame using a frozen model without query input. The default budget therefore contains 2,000 frame-caption pairs. A fixed decoding policy handles short or partially decodable clips, and duplicate frame timestamps are logged. Each branch takes the maximum query similarity across its four indexed items to produce one video score. All methods reuse the identical frame, caption, and feature caches.

**Runtime feasibility.** Before processing the full sample, a fixed 20-video subset drawn from the training partition measures loading, frame extraction, caption generation, and encoding time. Warm-up and model-download time are recorded separately. These videos retain their training assignment. The runtime estimate is T_setup + N * t_video + T_fusion/evaluation, where t_video is measured under the intended batch size. The estimate guides the budget but is not a promise of completion within one day. If the measured budget is unsuitable, a prespecified reduced tier uses 200 videos with 120/40/40 training/validation/test groups and the same four-frame rule. The tier is chosen before validation/test retrieval outcomes are inspected; the 200-video collection is a nested subset preserving the original group assignments. Results from different candidate-pool sizes are reported separately.

**Scope of evidence.** With 100 test-video groups in the default pilot, multiple descriptions increase query coverage but not the number of independent video groups. Confidence intervals therefore resample video groups. Conclusions are restricted to the sampled population and candidate size; full-dataset generalization, cross-domain robustness, and benchmark superiority remain untested. The previous broadcast collection may be evaluated separately once its data and relevance protocol are verified.

'''
source = source[:source.index('### 4.1 Dataset')] + dataset + source[source.index('### 4.2 Baselines'):]
source = source.replace('Full rankings are preferred for the 1,000-candidate pilot', 'Full rankings are used for the selected 500-video or 200-video pilot')
source = source.replace('full rankings for the 1,000-video candidate pool', 'full rankings for the selected pilot candidate pool')
source = source.replace('| Split seed | 42; saved manifest shared across methods |', '| Split seed | 42; video-group assignments shared across methods |\n| Default sample | 500 videos; 300/100/100 supervision groups |\n| Reduced tier | 200 videos; 120/40/40 supervision groups |\n| Frames per video | 4; one independently generated caption per frame |\n| Query cap | 5 distinct reference descriptions per video |')
source = re.sub(r'( The current design is a budgeted sampled pilot\.)+', '', source)
source = source.replace('Sections 1, 2, and 4 are drafted below in the requested outline.', 'Sections 1, 2, and 4 are drafted below in the requested outline. The current design is a budgeted sampled pilot.')
(BASE / 'PAPER_DRAFT_Sections_1_2_4.md').write_text(source, encoding='utf-8')

keys = {1:'radford2021clip',2:'luo2021clip4clip',3:'cormack2009rrf',4:'miech2018mee',5:'liu2019experts',6:'xu2016msrvtt'}

def inline(text):
    protected = []
    def hold(value):
        protected.append(value)
        return f'ZZPROTECTED{len(protected)-1}ZZ'
    text = re.sub(r'\\\(.*?\\\)', lambda m: hold(m[0]), text)
    text = re.sub(r'\[([1-6])\]', lambda m: hold(r'\cite{'+keys[int(m[1])]+'}'), text)
    replacements = {'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}
    text = ''.join(replacements.get(c,c) for c in text)
    for old,new in {'–':'--','—':'---','’':"'",'“':'``','”':"''",'…':r'\ldots{}','→':r'\(\rightarrow\)','∈':r'\(\in\)','·':r'\(\cdot\)'}.items():
        text = text.replace(old,new)
    text = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', text)
    for i,value in enumerate(protected): text = text.replace(f'ZZPROTECTED{i}ZZ',value)
    return text

def convert(content):
    lines=content.splitlines(); result=[]; i=0
    while i<len(lines):
        line=lines[i]
        if line.strip()==r'\[':
            block=[line]; i+=1
            while i<len(lines):
                block.append(lines[i]); i+=1
                if block[-1].strip()==r'\]': break
            result.extend(block); continue
        if line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].startswith('|'):
                cells=[c.strip() for c in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[-: ]+',c) for c in cells): rows.append(cells)
                i+=1
            n=len(rows[0]); cols='@{}'+('l' if n==4 else '')+'X'*(n-1 if n==4 else n)+'@{}'
            result += [r'\begin{center}',r'\small',r'\begin{tabularx}{\textwidth}{'+cols+'}',r'\toprule']
            for j,row in enumerate(rows):
                result.append(' & '.join(inline(c) for c in row)+r' \\')
                if j==0: result.append(r'\midrule')
            result += [r'\bottomrule',r'\end{tabularx}',r'\end{center}']; continue
        if line.startswith('### '):
            title=re.sub(r'^### \d+\.\d+ ', '',line)
            result.append(r'\subsection{'+inline(title)+'}')
        elif line.startswith('## '):
            title=re.sub(r'^## \d+ ', '',line)
            result.append(r'\section{'+inline(title)+'}')
        else: result.append(inline(line))
        i+=1
    return '\n'.join(result)+'\n'

body=source[source.index('## 1 Introduction'):source.index('## References')]
sections=re.split(r'(?=^## \d+ )',body,flags=re.M)
names=['01_introduction','02_related_work','03_methodology','04_experimental_setup','05_results','06_ablation_studies','07_conclusion']
(OUT/'sections').mkdir(exist_ok=True)
for name,section in zip(names,[s for s in sections if s.strip()]):
    (OUT/'sections'/f'{name}.tex').write_text(convert(section),encoding='utf-8')

main=r'''% Target venue: ACIIDS; checked against ACIIDS 2027 author guidelines.
% Standard length: 12--15 pages including references. See ACIIDS_AUTHOR_NOTES.md.
\documentclass[runningheads]{llncs}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb}
\usepackage{graphicx,booktabs,tabularx,array}
\usepackage[hidelinks]{hyperref}
\renewcommand{\UrlFont}{\ttfamily\small}
\setlength{\emergencystretch}{2em}
\title{Query-Adaptive Visual--Caption Fusion for Video Retrieval}
\titlerunning{Query-Adaptive Visual--Caption Fusion}
\author{Author names to be confirmed}
\authorrunning{Draft}
\institute{Affiliation to be confirmed}
\begin{document}
\maketitle
\begin{abstract}
Draft placeholder. The abstract will be completed after the method and experimental findings are established (target: 150--250 words).
\keywords{Video retrieval \and Multimodal fusion \and Query-adaptive weighting \and Generated captions}
\end{abstract}
\noindent\textit{Working draft: Sections 1, 2, and 4 are developed. Experimental settings describe a proposed sampled pilot, not completed runs.}

'''
main+='\n'.join(r'\input{sections/'+n+'}' for n in names)
main+=r'''
\bibliographystyle{splncs04}
\bibliography{references}
\end{document}
'''
(OUT/'main.tex').write_text(main,encoding='utf-8')
print('Generated main.tex and seven section files; synchronized Markdown pilot protocol.')
