# Questions for the Author

Only questions that cannot be answered reliably from the repository are included.

## Critical before drafting the paper

1. What university/program, capstone rubric, required learning outcomes, and exact course names/topics should be connected to this work?
Project – The capstone project is focused on the application of principles and the practice of engineering and is
not meant to be a mini-thesis. The capstone projects provide a mechanism to demonstrate mastery of concepts
learned in the program to a specific problem. Students can apply the skills and knowledge acquired in the program
to a known problem in order to develop an appropriate solution. These students could also work with faculty to
develop a solution to an issue faced in a lab or research group. The project report must be of sufficient length and
rigor to demonstrate this mastery. 
The project should be commensurate with a three-credit hour graduate course. Projects that include significant
data collection, extended collaborations, travel, and / or extensive analysis can be more than three credit hours
(this is the exception and only offered in certain programs).
• The project is not a thesis addressing a research issue. It is an application of knowledge and skills gained as part
of the Master of Engineering program.
• The project should demonstrate mastery of concepts and an application to a practical engineering or science
problem
Guidelines for the Project Report:
The final report for a capstone project must contain the following:
• Cover page (generated from Capstone Final Report form found in CEAS student portal), and the following in the
report:
• Abstract that succinctly describes the problem addressed, the methods used, and the results,
• Introduction that provides sufficient background to allow the reader to understand the problem, the constraints
and the relevant characteristics of the project,
• Methods (approach or analysis, as appropriate) that describe how the problem was addressed; this section
should provide some details on how the skills and knowledge gained through the MEng program contributed to
the solution,
• Results obtained through the project,
• Discussion of the efficacy of the approach, lessons learned through the project, areas for improvement,
additional work that could be performed, and
• Bibliography of references cited.
Project reports should not exceed 10 pages, double-spaced, 11-point font, and one-inch margins. Appendices with code
or graphs, for example, can be included and cited in the body of the report.
2. What was your individual contribution versus assistance from Codex, Claude, instructors, collaborators, or externally supplied code/data? How should AI-assisted development be disclosed?
Every idea and decision can be traced to me, but the code is ai assisted in development. maybe just mentioned once in methods.
3. Should the final narrative center on the evolution from the SPY intraday agent, or present the current multi-strategy platform as the primary artifact with SPY as its origin?
It should focus on evolution specificially the machine learning aspects and decisions
4. What was the capstone's formally approved problem statement and scope, and did it change after proposal/advisor review?
The orginal idea was the intraday swing trader on the SPY etf using xgboost, however that didnt perform well so after countless optimization, i moved on to the next problem, and the enxt, and they just grew.
5. Are real-money order submission or realized live results in scope for disclosure? If yes, which exact periods/accounts/results may be discussed without revealing private financial information?
no i havent used real money yet its still in testing
6. Have the historically committed Schwab credentials been revoked/rotated, and has Git history been purged? This determines whether the paper can describe the issue as remediated or only identified.
yes but thats so minor we dont disclose it. there were more important issues we overcame.
7. Which results do you want treated as the capstone's final evaluation cutoff? Should any data or experiments after a specific date be described as post-capstone follow-on work?
no, use everything you see fit.

## Important for accurate motivation and decision narrative

8. Which technical turning points were your deliberate decisions, and why: fuzzy setup labels, multiclass rather than dual models, cross-sectional ranking, rule confirmation, long-only HTF, shared execution, and dynamic themes?
everything was a deliberate decision. i think about it endlessly night and day or at least i used to in the first 8 months. so i was always thinking about ways to optimize it and to try new things. brainstorming. everything you mentioned there for sure. i was inspired by classwork, papers, news, trading myself. I would have an idea, then we would run experiments to determine if it would help or not. exactly the scientific method repeated a thousand times. adding theme rotation/expansion was definitely the biggest aha.
9. Which failed experiment most changed your understanding of AI engineering: leakage correction, weak triple-barrier/meta-entry results, paper option losses, theme shorts, catalyst data quality, selection-bias audit, confluence null result, or the WSL incident?
Definitely leakage correction and weak results. i was so surprised in reading other research papers how good the results were, but the more i read, the more i realized they never disclosed full test setups. I realized how easy it was to introduce leakage and skew results in your favor. it happened in like 4 or 5 different features we found so we had to retrain the models every time. In terms of weak results from certain models, yes that was frustrating. because to me it seems like a simple label, but it made me realize how noisy and fractal the stock market is. Its easy to get lost in the waves especially as you go to lower time frames because its like the butterfly effect. thats actually why we are trying to ride the waves of the big whales anyways. Also the PPO RL agent was a major flop. It was too difficult for the agent to learn using our test setups and hardware. Or maybe we didn't have neough good quality rich data or our actions/environment were too abstracted from reality. I am just one man.
10. What practical usage pattern was intended: personal decision support, unattended paper trading, eventual live automation, or a reusable research platform?
The whole time i was thinking i want an automated stock trader that removes emotions entirely. since i have always seen really good day traders i wanted to train an ai model to do the same thing. for spy it didnt work with the daytrading due to lack of information, but finding success on higher time frames. I knew it would take a lot of research but as you can see we learned a lot. 
11. What constraints came from hardware, data subscriptions/free tiers, API rate limits, cost, available time, or university requirements?
Mostly data. Options data specifically. We can easily get free daily candles and historical, but lower time frames, options, non delayed live data was rarely possible. hardware would be storage cause its a lot of data to store; that's why I want to move it to cloud storage. I am using alpaca free tier to stream the data, and using free apis. my plan was to get it somewhat working to make some money and then it would pay for itself but that hasnt worked out so far. We are still in the testing phase of every module with good promise.
12. Were there advisor decisions or feedback that directly caused a redesign? If so, provide dates or wording that may be paraphrased.
no dont mention.
13. What ethical/responsible-AI principles did you personally apply when deciding whether a model/result was safe to promote or deploy?
when it showed good performance on the test set which was kept separate the whole time. also using out of fold to avoid leakage, embarod boundaries, i also would test like 3 runs of training with different seeds and pick the best one to avoid just picking the single which could potentially be a higher or lower performance and i wouldnt know the relativety. we still dont with 3-5 but it is better. in terms of the global minima, i dont think there is one for the stock market. it is truly infinite. the only way to actually model it, would be simulate an entire new world with agents who participate in the stock market. its too complex for one mind to understand.

## Course and graduate-skill linkage

14. Which courses covered machine learning, deep learning, reinforcement learning, NLP/information retrieval, data engineering, software engineering, statistics/experimental design, optimization, visualization, cloud/distributed systems, or responsible AI?
I took deep learning which covered machine learning principles and models using colab. we learned a lot of tricks and architecture and how they work on the code/math scale. I took information retrieveal which taught me rag principles like embeddings and tokenization and similarity and clusters. retrieval and storage. also took applied ai, and applied gen ai which were both coding in colab courses ai problems. intelligent systems which was more of a biological view of artificial intelligence and industrial ai which reinforced good principles using machine learning. all classes were different amounts of overlapping with each other creating a sort of 3d venn diagram.  
15. Which specific course concepts did you consciously apply, rather than recognizing only afterward?
the idea to use news catalysts to create themes through embeddings and llm on the clusters was a combination of class concepts similar to rag. the optimizations and hyperparameter tuning on the models, training methods, and anywhere a random hyperparameter search could go. the data gather, preprocessing, pretty much the base pipeline for solving an ML problem was from class. but i made it my own version. 
16. Were any components originally course assignments or adapted from course-provided templates? If so, which ones?
no
17. Did the program require a particular engineering design process, risk analysis, testing standard, or reflection framework that should structure the report?
no

## Evaluation and reporting choices

18. May the paper use the July 2026 reproducibility audit and corrected frozen-test results as its primary evaluation, even if they postdate earlier advisor documents?
use any audit you want, just make sure to use teh most up to date one if its in a series. such as we had good results for a certain test, then discovered leakage, so we ran again with leakage removed and it performed worse but still good. include the most recent one in that case. cause i bet we have old results that are jsut the last time we ran that type of experiment. we probably ran a thousand experiments. 
19. Do you prefer the paper to report the later selective 4H policies (`final_balanced_k3_z2`, `final_ev_k5_z1_long`) or the deployed-winner clean baselines as primary? The dossier recommends showing both but treating the pre-test/frozen study as the stronger policy-selection design.
do the recommendation/ future change/implementation
20. Can the project provide a fixed historical universe definition or acquisition-date record, or must survivorship bias remain an acknowledged limitation?
its not relevent to the scope of this project, and for future ingestion we have multiple filters before adding to the universe. 
21. Are transaction costs, account size, option-contract assumptions, and paper/live account modes acceptable to state explicitly, or should the paper abstract some values for privacy?
just add the live implementation as intended for future experimentation.
22. Is there an approved benchmark beyond SPY buy-and-hold (for example equal-weight universe, sector-neutral selection, or cash/T-bill return)?
Module deployed (frozen test)	+92.5%, Sharpe 3.6, DD −15.2%	+336.0%, Sharpe 4.9, DD −18.7%
Random top-k, same engine+policy (10 seeds)	+28.5% ± 3.9	+31.5% ± 11.0
Equal-weight universe (module's own pool)	+64.7%, Sharpe 2.3	+64.3%
SPY buy & hold	+26.7%, Sharpe 1.8	+26.7%
Largest stock all-in (NVDA)	+45.5%, DD −20.2%	+56.2%
Sector-neutral (11 SPDR ETFs, equal wt)	+18.3%	+17.7%
3M T-bills (FRED DGS3MO accrual)	+4.5%
23. How should theme rotation be presented: as a substantive but explicitly in-sample/regime-sensitive legacy result, and as an architectural context module? Its historical rule-based equity curve substantially beat SPY, but the newer standalone theme-ML result is marginal and no theme-feature ablation yet quantifies its incremental contribution to meta-ranking.

it should be considered an oracle and what to aim our results for. but our ML didnt do well so its ponitless. we just added the theme metrics to meta ranker for it to add to its calculation.

## Writing and presentation inputs for the later phase

24. Please provide the writing samples, required citation style, template, page/word limit, font/spacing rules, figure/table rules, and submission deadline when authorizing the final-paper phase.
Writing samples will be provided as sources.
Abstract that succinctly describes the problem addressed, the methods used, and the results,
• Introduction that provides sufficient background to allow the reader to understand the problem, the constraints
and the relevant characteristics of the project,
• Methods (approach or analysis, as appropriate) that describe how the problem was addressed; this section
should provide some details on how the skills and knowledge gained through the MEng program contributed to
the solution,
• Results obtained through the project,
• Discussion of the efficacy of the approach, lessons learned through the project, areas for improvement,
additional work that could be performed, and
• Bibliography of references cited.
Project reports should not exceed 10 pages, double-spaced, 11-point font, and one-inch margins. Appendices with code
or graphs, for example, can be included and cited in the body of the report.
date is july 23rd 2026, the appendix and bibliography can go beyond the 10 page limit. apa style. 
25. Should the first-person voice be used, and may the system name “CynolycusBot” appear in the title?
third person only for this report as it should be formal in nature. The title should be targeted and academic yet concise. cynolycusbot can refer to the repo in the text but not in the title.
26. Which diagrams/figures best match the story you want to tell, and are screenshots containing tickers/account-adjacent information acceptable?
we should show relative performance, archistecture, pipeline, metrics, comparisions, backtest examples. 
27. Are appendices outside the page limit, and may the evidence index/results catalog be submitted as supplementary material?
Not as supplementary material, just use their information in the report. the appendices are outside the page limit.
28. Are there required references, textbooks, lectures, or advisor-provided papers that must appear in the bibliography?
use the sources as listed in /home/luket/repos/CynolycusBot/docs/InvestingResearchProject.docx which are research paper ideas. also list these sources of data (no need for descriptions). i also used openai and anthropic. Alpaca — equity bars, quotes, options data, streaming, asset universe, and broker/order APIs.
Schwab — option chains and optional market streaming for dealer positioning.
Polygon — legacy momentum-scalper minute bars and news.
Yahoo Finance / yfinance — news, earnings dates/history, company profiles/market cap, unusual options activity.
Finnhub — company news.
Google News RSS — aggregated financial news headlines.
SEC EDGAR — filings, XBRL/company facts, and 8-K exhibit text.
Financial Modeling Prep (FMP) — optional earnings-call transcripts.
CBOE — delayed option-chain snapshots / options context.
FINRA — daily short-sale volume.
NASDAQ — trading halts and short-interest data.
Federal Reserve — press-release RSS and published FOMC schedule.
FRED — Treasury yield series.
U.S. Treasury — fallback yield data.
OpenFDA — drug approvals.
ClinicalTrials.gov — trial updates.
USAspending.gov — federal contract awards.
TradingView — economic-calendar events.
PullPush — historical Reddit collection.
Reddit — public JSON and optional PRAW enrichment.
Anthropic Claude -  assists dynamic-theme taxonomy;
