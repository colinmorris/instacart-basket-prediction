from collections import namedtuple

Field = namedtuple('Field', 'name dtype')

ALL_FIELDS = [
    Field(*tup) for tup in [
      ('weight', float),
      ('previously_ordered', int), 
      ('lossmask', float),
      ('labels', float),

      ('seqlen', int),
      ]
]

id_fields = ['pid', 'aisleid', 'deptid', 'uid']
for idf in id_fields:
  f = Field(idf, int)
  ALL_FIELDS.append(f)

raw_feats = ['previously_ordered', 'days_since_prior', 'dow', 'hour', 'n_prev_products',] 
for rf in raw_feats:
  f = Field(rf, int)
  ALL_FIELDS.append(f)

FIELD_LOOKUP = {f.name: f for f in ALL_FIELDS}
