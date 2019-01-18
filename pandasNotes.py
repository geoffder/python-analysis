
'''
These are two ways of getting a desired subset of columns from a dataframe
with a hierarchy of columns. The method using list comprehension to build a
list of tuples that are used to index the columns is 3-4x faster (.4 vs 1.3s)
'''
# dirRecs = {c: {} for c in condition}
# for dend in manipSynsDF['dendNum']:
#     for i, c in enumerate(condition):
#         cols = [(tr, dir, rec) for tr in range(numTrials)
#                 for rec in dists[dend].index.values]
#         # dirRecs[c][dend] = treeRecs[i].loc[:, cols]
#         test = treeRecs[i].loc[:, cols]
#         dirRecs[c][dend] = treeRecs[i].drop(
#             columns=set(directions).difference([dir]),
#             level='direction'
#         ).drop(
#             columns=set(recInds).difference(dists[dend].index),
#             level='synapse'
#         )
#         assert(test.equals(dirRecs[c][dend]))
# return dirRecs
