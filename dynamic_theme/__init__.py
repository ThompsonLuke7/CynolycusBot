"""Dynamic Theme Taxonomy System.

Auto-discovers themes from ticker/news embeddings, labels them with Claude,
builds a relationship graph, and produces theme features for the Meta Ranker.

Daily  : update documents, embeddings, memberships, theme features
Weekly : recluster, run Claude labeling, discover new themes, update graph
"""
