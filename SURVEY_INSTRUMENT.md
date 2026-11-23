# Survey instrument

The practitioner survey of Section III-C (structure in Table II of the paper). It was
run in Portuguese with Google Forms. The question text below is quoted from the form
export (`survey_results.csv` in the data release, one column per question), followed by
an English translation. `analyze_survey.py` reads the export by column position; the
*Col.* column gives that position (0-based).

Single-choice options are listed as they appear in the responses: an option that no
respondent chose does not appear in the export. The multi-select questions (Q8, Q12)
have one follow-up column per option, so their option lists are complete.

## Consent

| Col. | Question | Type |
|---|---|---|
| 15 | *O Termo de Consentimento Livre e Esclarecido (TCLE) visa assegurar seus direitos como participante e está disponível [link omitted]. Por favor, leia com atenção e calma, buscando entender completamente a proposta da pesquisa. Não haverá qualquer tipo de penalização ou prejuízo se você não quiser participar ou retirar sua autorização em qualquer momento.* <br> The informed consent form protects your rights as a participant and is available [link]. Please read it carefully. There is no penalty if you do not take part or withdraw your consent at any time. | Yes / No (all 24 respondents answered Yes) |

The link to the consent form is omitted from the release.

## Section 1 – Demographics

| ID | Col. | Question | Options |
|---|---|---|---|
| Q1 | 32 | *Qual é o seu papel principal na área de desenvolvimento de software?* <br> What is your main role in software development? | Desenvolvedor(a) de Software (software developer); Cientista de Dados / Engenheiro(a) de IA (data scientist / AI engineer); Analista de Negócios (business analyst); Outro → *Qual?* (other, free text; col. 33) |
| Q2 | 2 | *Há quantos anos você possui experiência profissional na sua função principal?* <br> How many years of professional experience do you have in your main role? | 0-2 anos; 2-5 anos; 5-10 anos; 10+ anos |
| Q3 | 3 | *Qual é o seu modelo de trabalho atual?* <br> What is your current work model? | 100% presencial (on-site); Híbrido (hybrid); 100% remoto (remote) |
| Q4 | 4 | *Qual é o tamanho da sua equipe de desenvolvimento?* <br> How large is your development team? | 1-5; 6-10; 10-20; 20+ pessoas |
| Q5 | 5 | *Qual o tamanho da sua empresa?* <br> How large is your company? | Menos de 20; 20-100; 100-500; 500-2.000; Mais de 2.000 pessoas |
| Q6 | 6 | *Qual o modelo de negócio da empresa?* <br> What is the company's business model? | Produto próprio (SaaS, plataforma, app) (own product); Consultoria / software house / outsourcing; Fintech / banco / financeiro; Outro (other) |

## Section 2 – Productivity

| ID | Col. | Question | Type |
|---|---|---|---|
| Q7 | 7 | *Anote espontaneamente até 3 métricas que você considera boas quando pensa em medir produtividade de um desenvolvedor:* <br> Write down, unprompted, up to 3 metrics you consider good for measuring a developer's productivity. | Open text |
| Q8 | 8 | *Quais métricas sua empresa ou equipe usa para acompanhar produtividade? (marque todas que se aplicam)* <br> Which metrics does your company or team use to track productivity? (select all that apply) | Multi-select, options below |
| Q8a | 16–23 | For each selected metric: *Você considera que essa métrica representa bem a produtividade real do desenvolvedor?* <br> Do you think this metric represents the developer's real productivity well? <br> For *Outro*: *Qual métrica sua equipe usa? De 1 a 5, quão representativa você considera essa métrica?* (Which metric does your team use? From 1 to 5, how representative is it?) | Likert 1–5 (1 = does not represent, 5 = represents very well) |
| Q9 | 9 | *Na sua opinião, qual seria a métrica mais útil para medir a produtividade de um desenvolvedor - algo que ainda não é amplamente adotado?* <br> In your opinion, what would be the most useful metric for measuring a developer's productivity, something not yet widely adopted? | Open text |

Q8 options (follow-up column in brackets): Story points / Velocity (Scrum) [16];
Deployment frequency (quantas vezes por período é feito deploy) [17]; Change failure
rate / MTTR (DORA metrics) [18]; Throughput (número de tarefas/PRs entregues por
período) [19]; Code coverage / qualidade de código (ex: SonarQube) [20]; Performance
review / OKRs individuais [21]; Linhas de código / número de commits [22]; Outro [23];
Nenhuma métrica formal (no formal metric; no follow-up).

## Section 3 – Mental health

| ID | Col. | Question | Type |
|---|---|---|---|
| Q11 | 10 | *Sua empresa acompanha de alguma forma a saúde mental ou bem-estar dos desenvolvedores?* <br> Does your company track developers' mental health or well-being in any way? | Sim, de forma estruturada (ferramentas, pesquisas, programas formais) (yes, structured); Sim, de forma informal (1:1s, conversas com gestor) (yes, informal); Não acompanha (no); Não sei / nunca discutido (don't know / never discussed) |
| Q12 | 11 | *Quais práticas ou ferramentas são usadas? (marque todas que se aplicam)* <br> Which practices or tools are used? (select all that apply) | Multi-select, options below |
| Q12a | 24–31 | For each selected practice: *Você acredita que essa prática é efetiva para identificar e prevenir problemas de saúde mental em devs?* <br> Do you think this practice is effective at identifying and preventing mental-health problems in developers? <br> For *Outro*: *Quais? De 1 a 5, quão efetivas você acredita que essas práticas são?* (Which ones? From 1 to 5, how effective are they?) | Likert 1–5 (1 = not effective, 5 = very effective) |
| – | 12 | *Quais sinais ou indicadores você acha que melhor revelam que um desenvolvedor está com a saúde mental comprometida pelo trabalho? Pense em comportamentos, mudanças de padrão, qualidade do trabalho, etc.* <br> Which signals or indicators best reveal that a developer's mental health is being harmed by work? Think of behaviour, changes in patterns, quality of work, etc. | Open text (Section IV-D) |

Q12 options (follow-up column in brackets): Pulse surveys / pesquisas de clima (ex:
Office Vibe, Culture Amp) [24]; eNPS (Employee Net Promoter Score) [25]; 1:1s regulares
focados em bem-estar [26]; Programa de assistência ao empregado (EAP / psicólogo) [27];
Benefício de terapia / saúde mental [28]; "No meeting days" / proteção de tempo de foco
[29]; Nenhuma prática formal [30]; Outro [31].

## Section 4 – Holistic

| ID | Col. | Question | Type |
|---|---|---|---|
| Q14 | 13 | *O quanto você acredita que produtividade e saúde mental estão relacionadas no dia a dia de um desenvolvedor?* <br> How related do you think productivity and mental health are in a developer's day-to-day work? | Likert 1–5 (1 = completely independent, 5 = inseparable) |
| – | 14 | *Qual é o seu email? (opcional)* <br> Your e-mail (optional) | Open text; emptied in the data release |
